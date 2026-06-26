"""CAPT training losses (clean main = ITC + ROUTE/PACE).

The CAPT recipe (DSIP-deep / XDSIP backbone) trains with two losses:
  * ITC   : image-text contrastive (CoOp-style cross-entropy to the frozen,
            class-text anchors).
  * ROUTE/PACE : Instance-Prototype Routing (the headline novelty). Per-sample mixes
            CDCM (cross-domain contrastive) and CDPA (own-prototype pull) by the
            query's intrinsic clustering confidence w = cos(q, own-class proto):
                L_i = (1 - w_i) * CDCM_i + w_i * CDPA_i
            Well-clustered samples (w~1) get CDPA-only (tighten, no push -> keeps
            the seen boundary / UdCDR precision); poorly-clustered (w~0) get CDCM
            (separate -> Unseen/Mixed mAP). w (detached) is the SOLE mixing control,
            so ROUTE is parameter-free (no lambda / margin) and absorbs CDPA.

(Standalone CDCM/CDPA losses and the fixed-ratio router live only in the ablation
backup CaDiP/capt_dev/clasp_ablation_backup_20260616/ — used to regenerate the
component ablation table; not needed by the clean main recipe.)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def itc_loss(image_feat: torch.Tensor,
             text_feat: torch.Tensor,
             labels: torch.Tensor,
             temperature: float = 0.07) -> torch.Tensor:
    """Image-text contrastive (CoOp-style): cross-entropy of each prompted image
    feature against the frozen text-class anchors. image_feat (B,D) and
    text_feat (C,D) must both be L2-normalised."""
    logits = (image_feat @ text_feat.t()) / temperature   # (B, C) cosine / tau
    return F.cross_entropy(logits, labels.long())


def _cdcm_masks(query_classes, query_domains, bank_classes, bank_domains, sim, max_neg_sim):
    """Shared CDCM masks: cross-domain same-class positives (fallback same-class),
    cross-domain different-class negatives below ``max_neg_sim``."""
    sc_mask = bank_classes.unsqueeze(0) == query_classes.unsqueeze(1)        # same class
    dd_mask = bank_domains.unsqueeze(0) != query_domains.unsqueeze(1)        # diff domain
    cd_pos = sc_mask & dd_mask
    pos_mask = torch.where(cd_pos.any(dim=1, keepdim=True), cd_pos, sc_mask)  # cross-dom pos, fallback same-class
    neg_cls = ~sc_mask
    cd_neg = neg_cls & dd_mask
    neg_cls = torch.where(cd_neg.any(dim=1, keepdim=True), cd_neg, neg_cls)
    neg_mask = neg_cls & (sim < max_neg_sim)
    return pos_mask, neg_mask


def _cdcm_per_sample(logits, pos_mask, neg_mask, K_hard):
    """Multi-positive InfoNCE per query over positives + top-K hardest negatives."""
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    neg_logits = logits.masked_fill(~neg_mask, float("-inf"))
    K = min(K_hard, neg_logits.size(1))
    topk_neg, _ = neg_logits.topk(K, dim=1)                                 # hardest negatives
    denom = torch.logsumexp(torch.cat([pos_logits, topk_neg], dim=1), dim=1)
    n_pos = pos_mask.sum(dim=1).clamp(min=1)
    mean_pos = torch.where(pos_mask, logits, torch.zeros_like(logits)).sum(dim=1) / n_pos
    return -(mean_pos - denom)                                             # (B,) per-sample CDCM


def route_loss(query_feats, query_classes, query_domains,
               bank_feats, bank_classes, bank_domains,
               tau=0.07, K_hard=256, max_neg_sim=0.9,
               soft=0.0, gap=0):
    """PACE — Per-sample Alignment via Domain-gap Estimation (CAPT headline N2).

    Per-sample mixes CDCM (cross-domain contrastive push) and CDPA (own-prototype
    pull):

        L_i = (1 - w_i) * CDCM_i + w_i * CDPA_i

    w_i ~ 1 -> CDPA-only (tighten, protects seen-class / UdCDR precision);
    w_i ~ 0 -> CDCM (cross-domain separation -> Unseen/Mixed mAP).

    Router weight w (detached):
      * ``gap >= 1`` (default): route by the RESIDUAL DOMAIN GAP
          g_i = cos(q, same-domain proto) - cos(q, cross-domain proto).
          gap==2 (default DG-PACE): combine the standardized confidence and gap,
            w = sigmoid(z(s) - z(g));   gap==1: pure gap, w = sigmoid(-(g - mean)/soft).
          Samples still bound to their source domain (large g) get CDCM; already
          domain-invariant ones (g~0) get CDPA. Falls back to confidence-only when a
          sample's same-domain prototype is missing from the bank.
      * ``gap == 0``: legacy PACE — confidence only, w = sigmoid((cos - mean)/soft).
    All features L2-normalised internally. Scalar (0 if the bank is empty)."""
    if bank_feats.numel() == 0:
        return query_feats.sum() * 0.0
    q = F.normalize(query_feats, dim=-1)
    bank_n = F.normalize(bank_feats.to(q.dtype), dim=-1)

    # own-class domain-averaged prototype: shared by the CDPA term AND the router
    same = (bank_classes.unsqueeze(0) == query_classes.unsqueeze(1)).to(q.dtype)   # (B, N)
    cnt = same.sum(dim=1)
    keep = cnt > 0
    if keep.sum() == 0:
        return query_feats.sum() * 0.0
    proto = F.normalize((same @ bank_n) / cnt.clamp(min=1).unsqueeze(1), dim=-1)   # (B, D)
    satg = (q * proto).sum(dim=-1)                                                 # cos to own proto (grad thru q)
    cdpa_i = (1.0 - satg).clamp(min=0.0)                                           # CDPA pull, per-sample

    # per-sample CDCM (multi-positive InfoNCE); same masks as cdcm_loss
    sim = q @ bank_n.t()
    pos_mask, neg_mask = _cdcm_masks(query_classes, query_domains,
                                     bank_classes, bank_domains, sim, max_neg_sim)
    cdcm_i = _cdcm_per_sample(sim / tau, pos_mask, neg_mask, K_hard)
    cdcm_valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)                         # CDCM undefined -> drop push
    cdcm_i = torch.where(cdcm_valid, cdcm_i, torch.zeros_like(cdcm_i))

    # per-sample routing (detached weight; no per-batch rescale)
    sd = satg.detach()
    if gap > 0:
        # DG-PACE: residual domain gap g = cos(q, same-dom proto) - cos(q, cross-dom proto)
        same_dom = same * (bank_domains.unsqueeze(0) == query_domains.unsqueeze(1)).to(q.dtype)
        cnt_s = same_dom.sum(dim=1)
        has_s = cnt_s > 0
        proto_s = F.normalize((same_dom @ bank_n) / cnt_s.clamp(min=1).unsqueeze(1), dim=-1)
        g = ((q * proto_s).sum(dim=-1) - satg).detach()                           # large g -> still domain-bound
        gbar = g[has_s].mean() if bool(has_s.any()) else g.mean()
        gstd = (g[has_s].std() if bool(has_s.any()) else g.std()) + 1e-6
        s_eff = soft if soft > 0 else 0.1
        if gap >= 2:                                                              # combine confidence + gap (standardized)
            sd_z = (sd - sd.mean()) / (sd.std() + 1e-6)
            w_gap = torch.sigmoid(sd_z - (g - gbar) / gstd)
        else:                                                                     # pure domain-gap routing
            w_gap = torch.sigmoid(-(g - gbar) / s_eff)
        w_conf = torch.sigmoid((sd - sd.mean()) / soft) if soft > 0 else sd.clamp(0.0, 1.0)
        w = torch.where(has_s, w_gap, w_conf)                                     # fallback to confidence if no same-dom proto
    elif soft > 0.0:
        w = torch.sigmoid((sd - sd.mean()) / soft)                                # legacy PACE: confidence soft-threshold
    else:
        w = sd.clamp(0.0, 1.0)                                                     # raw cosine
    L = (1.0 - w) * cdcm_i + w * cdpa_i
    return L[keep].mean()
