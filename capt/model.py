"""CAPT — main model (XDSIP + PACE, clean main path with ablation branches removed).

   The actual main file imported by train.py. All ablation flags were stripped so
   only the main path remains; the forward pass and parameter count were verified
   bit-identical (max|delta|=0.0) against the previous ablation-inclusive version
   (2026-06-16). To reproduce ablations, use the backup
   CaDiP/capt_dev/capt_ablation_backup_20260616/.
   The generator always runs as "frozen text anchors + instance-conditional".

[One-line summary]
   On top of a frozen CLIP ViT-B/32, generate a visual prompt "per image" and inject
   it into all 12 layers. The prompt is built from the frozen CLIP 'text class anchors'
   lightly corrected by the image (= XDSIP). Only ~1.1-1.35M trainable params (backbone
   frozen). Retrieval is cosine similarity between f_prompted features.

forward(image) -> {"f_prompted": (B,512) image feature, "text_tune": (C,512) class-text feature}
   both are L2-normalised.
"""
from __future__ import annotations

from typing import List

import clip as clip_pkg
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_frozen_clip(backbone: str, device):
    # Load CLIP and freeze every parameter (no training). The backbone is fixed; we only train prompts.
    model, _ = clip_pkg.load(backbone, device=device, jit=False)
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class CAPTNet(nn.Module):
    def __init__(self, args, class_names: List[str],
                 domain_names: List[str], device):
        super().__init__()
        self.device = device
        self.clip = _load_frozen_clip(getattr(args, "clip_backbone", "ViT-B/32"), device)
        d_v = int(self.clip.visual.conv1.out_channels)        # vision token dim 768
        d_out = int(self.clip.text_projection.shape[1])       # output (shared embedding) dim 512
        D_emb = self.clip.token_embedding.embedding_dim       # text token embedding dim 512
        self.d_v, self.d_out = d_v, d_out

        # ---- frozen text class anchors: encode "a photo of a {class}" with frozen CLIP ----
        #      these anchors act as (1) the basis for prompt generation, (2) the target of the ITC loss.
        with torch.no_grad():
            tok = torch.cat([clip_pkg.tokenize(f"a photo of a {c.replace('_',' ')}")
                             for c in class_names]).to(device)
            text_clip = F.normalize(self.clip.encode_text(tok).float(), dim=-1)
        self.register_buffer("text_clip", text_clip)          # (C, 512) frozen (buffer, not trained)
        self.delta_t = nn.Parameter(torch.zeros(text_clip.shape))  # output correction: per-class text shift (starts at 0)

        # ---- input-level text context (CoOp-style, learnable) ----
        #      learn the 4 X tokens in "[X][X][X][X] {class}" to lightly tune the text encoder 'input'.
        M_cls = int(getattr(args, "text_ctx_cls", 4))
        self.n_ctx = M_cls
        self.text_ctx_cls = nn.Parameter(torch.randn(M_cls, D_emb) * 0.02)  # 0.02 = CoOp standard init (start small)
        placeholder = " ".join(["X"] * self.n_ctx)
        prompts = [f"{placeholder} {c.replace('_',' ')}" for c in class_names]
        with torch.no_grad():
            ctok = torch.cat([clip_pkg.tokenize(p) for p in prompts]).to(device)
        self.register_buffer("class_tokens", ctok, persistent=False)
        self.register_buffer("eot_idx", ctok.argmax(dim=-1), persistent=False)  # EOT (end) token position of each sentence

        # ---- XDSIP: text-anchor based 'image-conditional' deep prompt generator (core novelty) ----
        Pq = int(getattr(args, "n_prompt", 8))   # number of visual prompt tokens to generate (used as a whole)
        r = int(getattr(args, "rank", 64))       # low-rank bottleneck (saves compute / params)
        n_heads = int(getattr(args, "gen_heads", 4))
        self.n_prompt, self.rank = Pq, r
        n_blocks = len(self.clip.visual.transformer.resblocks)
        # layers to inject the prompt into: empty = all 0~11 (deep), "0" = input layer only (shallow ablation)
        layers = str(getattr(args, "inject_layers", ",".join(str(i) for i in range(n_blocks))))
        self.inject_layers = sorted({int(s) for s in layers.split(",") if s.strip() != ""
                                     if 0 <= int(s) < n_blocks})
        assert 0 in self.inject_layers, "must inject at layer 0 (input)"

        self.q_base = nn.Parameter(torch.randn(Pq, r) * 0.02)  # learnable base queries for the Pq prompts
        self.q_proj = nn.Linear(d_v, r)                       # current CLS summary -> query (adds image conditioning)
        self.kv_proj = nn.Linear(D_emb, r)                    # text anchor -> r dim (Key/Value)
        self.kv_patch = nn.Linear(d_v, r)                     # current patch tokens -> r dim
        self.xgate = nn.Parameter(torch.tensor(-3.0))         # text<-vision correction gate (sigmoid(-3)~0.05, start small)
        self.attn = nn.ModuleDict()  # per-layer cross-attention
        self.up = nn.ModuleDict()    # per-layer r -> d_v up-projection
        for L in self.inject_layers:
            self.attn[str(L)] = nn.MultiheadAttention(r, n_heads, batch_first=True)
            up = nn.Linear(r, d_v)
            nn.init.trunc_normal_(up.weight, std=0.02); nn.init.zeros_(up.bias)  # start near 0 (almost no intervention at first)
            self.up[str(L)] = up
        for m in (self.q_proj, self.kv_proj, self.kv_patch):
            nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

        # ---- output feature refine head (absorbs the QE role into the DSIP output; residual MLP) ----
        self.use_outhead = int(getattr(args, "dsip_outhead", 0))     # 1 = use it (headline method)
        if self.use_outhead:
            rdim = int(getattr(args, "refine_dim", 256))             # bottleneck width
            self.out_head = nn.Sequential(nn.Linear(d_out, rdim), nn.GELU(), nn.Linear(rdim, d_out))
            nn.init.zeros_(self.out_head[-1].weight); nn.init.zeros_(self.out_head[-1].bias)  # start at 0 (residual)

        # ---- cross-domain feature bank for CDCM (FIFO ring buffer): the prototype store PACE routing reads from ----
        N = int(getattr(args, "bank_size", 10000))
        self.register_buffer("cdq_features", torch.zeros(N, d_out), persistent=False)  # stored features
        self.register_buffer("cdq_classes", torch.full((N,), -1, dtype=torch.long), persistent=False)  # class labels
        self.register_buffer("cdq_domains", torch.full((N,), -1, dtype=torch.long), persistent=False)  # domain labels
        self.register_buffer("cdq_valid", torch.zeros(N, dtype=torch.bool), persistent=False)  # whether filled
        self.register_buffer("cdq_ptr", torch.zeros(1, dtype=torch.long), persistent=False)  # next write position

        self.float()  # generator / text-context are fp32 (CLIP is already float)
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[CAPT] XDSIP inject_layers={self.inject_layers} Pq={Pq} rank={r}; "
              f"trainable={n_train:,} (frozen CLIP {getattr(args,'clip_backbone','ViT-B/32')})")

    # ------------------------------------------------------------------
    def _gen_prompt(self, L: int, cls_summary: torch.Tensor, patches: torch.Tensor):
        """Language-anchored prompt read-out (single cross-attention).

        *Reads out* the visual prompt from the frozen CLIP 'text-class anchors':
        learnable queries cross-attend to the anchors to obtain Pq prompt tokens.
        The anchors are first lightly image-conditioned by the current patches (gated),
        so the prompt is instance-aware while staying on the language manifold.
        (Not a separated step — a single read-out.)
        """
        B = cls_summary.shape[0]
        scale = self.kv_proj.out_features ** -0.5
        # frozen language anchors -> r-dim key/value
        anchor = self.kv_proj(self.text_clip.float()).unsqueeze(0).expand(B, -1, -1)    # (B,C,r)
        patch = self.kv_patch(patches.float())                                          # (B,P,r)
        # lightly image-condition the anchors with the patches (gated residual) — each anchor shifts toward its matching patches
        img_cond = torch.bmm(torch.softmax(anchor @ patch.transpose(1, 2) * scale, dim=-1), patch)
        anchor = anchor + torch.sigmoid(self.xgate) * img_cond                          # (B,C,r)
        # static learnable queries read the prompt out of the anchors (paper default: instance_cond=0).
        q = self.q_base.unsqueeze(0)
        if int(getattr(self, "instance_cond", 0)):                                       # optional per-image query (not adopted in the paper)
            query = q + self.q_proj(cls_summary.float()).unsqueeze(1)                     # (B,Pq,r)
        else:
            query = q.expand(cls_summary.shape[0], -1, -1)                               # (B,Pq,r) static
        prompt, _ = self.attn[str(L)](query, anchor, anchor)                            # (B,Pq,r)
        return self.up[str(L)](prompt)                                                  # (B,Pq,d_v)

    def _forward_visual(self, image: torch.Tensor):
        # Run the frozen CLIP vision encoder, inserting a generated prompt at each specified layer.
        v = self.clip.visual
        x = v.conv1(image.type(self.clip.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)        # (B,P,d_v) patch tokens
        B = x.shape[0]
        cls = v.class_embedding.to(x.dtype) + torch.zeros(
            B, 1, x.shape[-1], dtype=x.dtype, device=x.device)           # CLS token
        cls = cls + v.positional_embedding[0:1, :].to(x.dtype)            # positional embedding (CLS)
        patches = x + v.positional_embedding[1:, :].to(x.dtype)          # positional embedding (patches)
        Pq = self.n_prompt
        # generate the input-layer (layer 0) prompt, then concatenate as [CLS, prompt, patches]
        P0 = self._gen_prompt(0, cls.squeeze(1), patches).to(patches.dtype)
        seq = v.ln_pre(torch.cat([cls, P0, patches], dim=1)).permute(1, 0, 2)  # LND order
        blocks = list(v.transformer.resblocks)
        for i, blk in enumerate(blocks):
            seq = blk(seq)                                               # pass through one transformer block
            nl = i + 1
            if nl in self.inject_layers and nl < len(blocks):
                # regenerate the next-layer prompt from the 'current' CLS/patches and replace it (deep, refreshed every layer)
                P_L = self._gen_prompt(nl, seq[0], seq[1 + Pq:].permute(1, 0, 2)).to(seq.dtype)
                seq = torch.cat([seq[0:1], P_L.permute(1, 0, 2), seq[1 + Pq:]], dim=0)
        cls_f = (v.ln_post(seq[0]) @ v.proj).float()                     # project only the CLS token into the shared space (512)
        return cls_f / cls_f.norm(dim=-1, keepdim=True)                  # L2 normalise -> image feature

    def _encode_text(self) -> torch.Tensor:
        # Text encoding: in "[X..X] {class}", replace the X slots (after BOS) with the learnable text_ctx_cls and run the frozen encoder.
        dt = self.clip.dtype
        x = self.clip.token_embedding(self.class_tokens).type(dt)
        C = x.shape[0]
        ctx = self.text_ctx_cls
        ctx = ctx.unsqueeze(0).expand(C, -1, -1).type(dt)
        x = torch.cat([x[:, :1], ctx, x[:, 1 + self.n_ctx:]], dim=1)     # [BOS, learnable context, class tokens...]
        x = x + self.clip.positional_embedding.type(dt)
        x = self.clip.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)   # frozen text transformer
        x = self.clip.ln_final(x).type(dt)
        x = x[torch.arange(C, device=x.device), self.eot_idx] @ self.clip.text_projection  # EOT token -> class feature
        return F.normalize(x.float(), dim=-1)

    def forward(self, image: torch.Tensor) -> dict:
        # Main forward: returns the (prompted) image feature + the class-text feature.
        f = self._forward_visual(image)                                 # prompted image feature
        if self.use_outhead:                                            # refine head (if present, residual refinement)
            f = F.normalize(f + self.out_head(f), dim=-1)
        text_tune = F.normalize(self._encode_text() + self.delta_t, dim=-1)  # text = input context + output shift
        return {"f_prompted": f, "text_tune": text_tune}

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor, classes: torch.Tensor, domains: torch.Tensor):
        # Normalise the batch features and push them into the bank (ring buffer). Overwrites the oldest slot (FIFO).
        feats = F.normalize(feats, dim=-1).float()
        B, N = feats.shape[0], self.cdq_features.shape[0]
        idx = (torch.arange(B, device=feats.device) + int(self.cdq_ptr.item())) % N
        self.cdq_features[idx] = feats
        self.cdq_classes[idx] = classes.to(self.cdq_classes.dtype)
        self.cdq_domains[idx] = domains.to(self.cdq_domains.dtype)
        self.cdq_valid[idx] = True
        self.cdq_ptr[0] = (int(self.cdq_ptr.item()) + B) % N

    def bank(self):
        # Return only the currently filled (valid) bank entries (features, classes, domains). Used by the losses (CDCM/PACE).
        v = self.cdq_valid
        return self.cdq_features[v], self.cdq_classes[v], self.cdq_domains[v]
