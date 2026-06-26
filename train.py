"""CAPT — train & evaluate for Universal Cross-Domain Retrieval (UCDR / UdCDR).

CAPT = DSIP (Deep Self-Instance Prompting) + PACE (Per-sample Alignment via
Cross-domain gap Estimation) on a frozen CLIP ViT-B/32 backbone.
The default flags below reproduce the paper's CAPT-deep results.

Example (DomainNet, holdout = sketch):
  python train.py -dataset_path /path/to/datasets \
      -seen_domain quickdraw -holdout_domain sketch -gallery_domain real \
      -epochs 4 -batch_size 60
"""
import argparse
import os
import random

import numpy as np
import torch

from capt.model import CAPTNet
from capt.engine import build_data, train_one_epoch, run_eval


def get_args():
    p = argparse.ArgumentParser("CAPT")
    # data
    p.add_argument("-dataset", default="DomainNet", choices=["DomainNet", "Sketchy", "TUBerlin"])
    p.add_argument("-dataset_path", required=True, help="root holding <dataset>/... images")
    p.add_argument("-is_eccv_split", type=int, default=1,
                   help="Sketchy only: 1 = ECCV'18 unseen split (matches reported numbers), 0 = random split.")
    p.add_argument("-seen_domain", default="quickdraw")
    p.add_argument("-holdout_domain", default="sketch")
    p.add_argument("-gallery_domain", default="real")
    p.add_argument("-include_auxillary_domains", type=int, default=1)
    p.add_argument("-trainvalid", type=int, default=0)
    # backbone / prompt
    p.add_argument("-clip_backbone", default="ViT-B/32")
    p.add_argument("-n_prompt", type=int, default=8,
                   help="number of generated visual prompt tokens Pq.")
    p.add_argument("-rank", type=int, default=64)
    p.add_argument("-gen_heads", type=int, default=4)
    p.add_argument("-inject_layers", default="",
                   help="comma layer ids for prompt injection; empty = all 12 (deep). '0' = shallow.")
    p.add_argument("-text_ctx_cls", type=int, default=4)
    # losses: L = ITC + lambda_route * PACE  (instance-prototype routing, parameter-free)
    p.add_argument("-temperature", type=float, default=0.07)
    p.add_argument("-route_soft", type=float, default=0.1,
                   help="PACE-B soft-threshold w = sigmoid((cos - batch_mean)/tau), tau = this value.")
    p.add_argument("-lambda_route", type=float, default=0.5,
                   help="weight on the PACE term (L = ITC + lambda_route*PACE).")
    p.add_argument("-pace_gap", type=int, default=2,
                   help="DG-PACE routing signal: 2 = confidence + residual domain gap (default, "
                        "headline); 1 = pure domain gap; 0 = legacy confidence-only PACE.")
    p.add_argument("-dsip_outhead", type=int, default=1,
                   help="residual MLP refine head on the DSIP output feature (part of the method).")
    p.add_argument("-refine_dim", type=int, default=256,
                   help="refine-head bottleneck width (256 default ~0.26M).")
    p.add_argument("-cdcm_tau", type=float, default=0.07)
    p.add_argument("-cdcm_K", type=int, default=256)
    p.add_argument("-cdcm_max_neg_sim", type=float, default=0.9)
    p.add_argument("-bank_size", type=int, default=10000)
    # optimisation
    p.add_argument("-epochs", type=int, default=4)
    p.add_argument("-batch_size", type=int, default=60)
    p.add_argument("-lr", type=float, default=1e-4)
    p.add_argument("-weight_decay", type=float, default=1e-4)
    p.add_argument("-num_workers", type=int, default=3)
    p.add_argument("-image_size", type=int, default=224)
    p.add_argument("-randaug_n", type=int, default=2)
    p.add_argument("-randaug_m", type=int, default=9)
    p.add_argument("-seed", type=int, default=0)
    p.add_argument("-save_dir", default="checkpoints")
    return p.parse_args()


def main():
    args = get_args()
    if not args.inject_layers:
        args.inject_layers = ",".join(str(i) for i in range(12))   # deep (default CAPT)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = build_data(args)
    model = CAPTNet(args, data["classnames"], data["domainnames"], device).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)  # constant LR
    print(f"[CAPT] trainable params: {sum(p.numel() for p in params):,}", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        print(f"\n=== epoch {epoch + 1}/{args.epochs} ===", flush=True)
        train_one_epoch(model, data["train_loader"], optimizer, args,
                        data["dict_clss"], data["dict_doms"], device)
        print(f"*** eval epoch {epoch + 1} ***", flush=True)
        te_data, sel = run_eval(model, args, data, device)
        print(f"ep {epoch + 1} selection mAP = {sel:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)},
                   os.path.join(args.save_dir, f"capt_{args.holdout_domain}_ep{epoch+1}.pt"))


if __name__ == "__main__":
    main()
