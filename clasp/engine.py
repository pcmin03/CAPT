"""CAPT training / evaluation engine on DomainNet (ECCV / Fang'24 split).

Reuses the standard UCDR data pipeline (shared with ProS/DePro/UCDR-Adapter)
under ``data/`` and the retrieval metrics under ``clasp/metrics.py``. The
training step is the clean CAPT recipe:
    L = ITC + lambda_route * PACE   (instance-prototype routing; parameter-free).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T

from data.DomainNet import domainnet
from data.Sketchy import sketchy_extended
from data.TUBerlin import tuberlin_extended
from data.dataloaders import CuMixloader, BaselineDataset
from data.sampler import BalancedSampler
from clasp.metrics import compute_retrieval_metrics
from clasp.utils import numeric_classes, create_dict_texts
from clasp.losses import itc_loss, route_loss

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transforms(image_size=224, randaug_n=2, randaug_m=9):
    return {
        "train": T.Compose([
            T.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.RandAugment(num_ops=randaug_n, magnitude=randaug_m),
            T.ToTensor(), T.Normalize(_CLIP_MEAN, _CLIP_STD)]),
        "eval": T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(), T.Normalize(_CLIP_MEAN, _CLIP_STD)]),
    }


def build_data(args):
    """Returns a dict with train_loader, class/domain dicts, splits, transforms."""
    if args.dataset == "Sketchy":
        di = sketchy_extended.create_trvalte_splits(args)
    elif args.dataset == "TUBerlin":
        di = tuberlin_extended.create_trvalte_splits(args)
    else:
        di = domainnet.create_trvalte_splits(args)
    tr_classes, va_classes, te_classes = di["tr_classes"], di["va_classes"], di["te_classes"]
    splits = di["splits"]
    dict_clss = create_dict_texts(tr_classes)
    te_dict_class = create_dict_texts(tr_classes + va_classes + te_classes)
    fls_tr = splits["tr"]
    dom_tr = np.array([f.split("/")[-3] for f in fls_tr])
    cls_tr = np.array([f.split("/")[-2] for f in fls_tr])
    tr_domains = np.unique(dom_tr).tolist()
    dict_doms = create_dict_texts(tr_domains)
    domain_ids = numeric_classes(dom_tr, dict_doms)
    tfm = build_transforms(args.image_size, args.randaug_n, args.randaug_m)
    data_train = CuMixloader(fls_tr, cls_tr, dom_tr, dict_doms, transforms=tfm["train"])
    sampler = BalancedSampler(domain_ids, args.batch_size // max(1, len(tr_domains)),
                              domains_per_batch=len(tr_domains))
    train_loader = DataLoader(data_train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    return dict(train_loader=train_loader, dict_clss=dict_clss, dict_doms=dict_doms,
                te_dict_class=te_dict_class, tr_classes=tr_classes, va_classes=va_classes,
                te_classes=te_classes, tfm=tfm, splits=splits,
                classnames=list(dict_clss.keys()), domainnames=list(dict_doms.keys()))


def train_one_epoch(model, loader, optimizer, args, dict_clss, dict_doms, device):
    model.train()
    for it, (im, cls_list, dom_list) in enumerate(
            tqdm(loader, desc="train", leave=False)):
        im = im.float().to(device, non_blocking=True)
        labels = torch.tensor(numeric_classes(cls_list, dict_clss),
                              dtype=torch.long, device=device)
        domains = torch.tensor(numeric_classes(dom_list, dict_doms),
                               dtype=torch.long, device=device)
        out = model(im)
        f, text = out["f_prompted"], out["text_tune"]
        # CAPT recipe: L = ITC + lambda_route * PACE (instance-prototype routing, parameter-free)
        l_itc = itc_loss(f, text, labels, args.temperature)
        bf, bc, bd = model.bank()
        l_route = route_loss(f, labels, domains, bf, bc, bd,
                             args.cdcm_tau, args.cdcm_K, args.cdcm_max_neg_sim,
                             soft=getattr(args, "route_soft", 0.0),
                             gap=getattr(args, "pace_gap", 2))
        loss = l_itc + getattr(args, "lambda_route", 0.5) * l_route
        if it % 200 == 0:
            print(f"  [train] it {it}  itc {l_itc.item():.3f}  "
                  f"route {float(l_route):.3f}", flush=True)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        # bank stores the refined feature (PACE routes on a refined prototype bank).
        model.enqueue(f.detach(), labels, domains)


@torch.no_grad()
def _embed(model, loader, te_dict_class, device, tag=""):
    model.eval()
    feats, labs = [], []
    for im, cls_list, _dom in tqdm(loader, desc=f"embed-{tag}", leave=False):
        im = im.float().to(device, non_blocking=True)
        feats.append(model(im)["f_prompted"].cpu())
        labs.append(torch.tensor(numeric_classes(cls_list, te_dict_class), dtype=torch.long))
    return torch.cat(feats, 0), torch.cat(labs, 0)


@torch.no_grad()
def evaluate(model, q_loader, g_loader, te_dict_class, device, tag=""):
    qf, ql = _embed(model, q_loader, te_dict_class, device, f"{tag}-q")
    gf, gl = _embed(model, g_loader, te_dict_class, device, f"{tag}-g")
    return compute_retrieval_metrics(qf.to(device), ql.to(device),
                                     gf.to(device), gl.to(device))


def run_eval(model, args, data, device):
    """The 5 DomainNet protocols: (sd,G0)(sd,G1)(hd,G0)(hd,G1) + UdCDR(seen).
    Returns list of metric dicts; index 3 = held-out Mixed-gallery (selection)."""
    tr_classes, va_classes, te_classes = data["tr_classes"], data["va_classes"], data["te_classes"]
    tfm, te_dict_class = data["tfm"], data["te_dict_class"]

    if args.dataset in ("Sketchy", "TUBerlin"):
        # ZS-SBIR: a single sketch->image retrieval over the unseen (test) classes.
        sp = data["splits"]
        ql = DataLoader(BaselineDataset(sp["query_te"], transforms=tfm["eval"]),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        gl = DataLoader(BaselineDataset(sp["gallery_te"], transforms=tfm["eval"]),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        print(f"[{args.dataset}] sketch->image  "
              f"#q={len(sp['query_te'])} #g={len(sp['gallery_te'])}", flush=True)
        m = evaluate(model, ql, gl, te_dict_class, device, tag=args.dataset)
        print(f"  mAP@200 = {m['mAP@200']:.4f}  prec@200 = {m['prec@200']:.4f}  "
              f"mAP@all = {m['mAP@all']:.4f}  prec@100 = {m['prec@100']:.4f}", flush=True)
        print(f"selection mAP = {m['mAP@200']:.4f}", flush=True)
        return [m], m["mAP@200"]

    specs = []
    for dom in [args.seen_domain, args.holdout_domain]:
        for gen in [0, 1]:
            specs.append(("UCDR", dom, args.gallery_domain, gen))
    specs.append(("UdCDR", args.holdout_domain, args.gallery_domain, "seen_only"))

    te_data = []
    for protocol, qdom, gdom, gen in specs:
        if protocol == "UdCDR":
            p = 0.1 if args.holdout_domain == "quickdraw" else 0.25
            q_files = np.array(domainnet.seen_cls_te_samples(args, qdom, tr_classes, p))
            g_files = np.array(domainnet.seen_cls_te_samples(args, gdom, tr_classes, p))
        else:
            qs = domainnet.trvalte_per_domain(args, qdom, 0, tr_classes, va_classes, te_classes)
            gs = domainnet.trvalte_per_domain(args, gdom, gen, tr_classes, va_classes, te_classes)
            q_files, g_files = np.array(qs["te"]), np.array(gs["te"])
        ql = DataLoader(BaselineDataset(q_files, transforms=tfm["eval"]),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        gl = DataLoader(BaselineDataset(g_files, transforms=tfm["eval"]),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        print(f"[{protocol}] Query:{qdom}; Gallery:{gdom}; Generalized:{gen}", flush=True)
        m = evaluate(model, ql, gl, te_dict_class, device, tag=f"{protocol}_{qdom}-G{gen}")
        te_data.append(m)
        print(f"  mAP@200 = {m['mAP@200']:.4f}  prec@200 = {m['prec@200']:.4f}", flush=True)
    sel_idx = 3   # held-out Mixed-gallery mAP@200 (selection metric)
    sel = te_data[sel_idx]["mAP@200"] if len(te_data) > sel_idx else (te_data[0]["mAP@200"] if te_data else 0.0)
    print(f"selection mAP = {sel:.4f}", flush=True)
    return te_data, sel
