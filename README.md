# CAPT — Cross-modal Anchored Prompt Tuning

Reference implementation of **CAPT** for **Universal Cross-Domain Retrieval
(UCDR / U^dCDR)** on a **frozen CLIP ViT-B/32** backbone (~1.35M trainable
parameters). CAPT consists of two components:

- **DSIP — Deep Self-Instance Prompting.** At every transformer layer, DSIP
  image-conditions the frozen CLIP text-class anchors and a set of *static*
  learnable queries read out an instance-specific visual prompt from them. The
  prompt is injected at all 12 layers; the backbone stays frozen.
- **PACE — Per-sample Alignment via Cross-domain gap Estimation.** A FIFO
  feature bank supplies two complementary objectives — **CDCM** (cross-domain
  contrastive matching, *spread*) and **CDPA** (class-prototype anchoring,
  *tighten*) — mixed **per sample** by a parameter-free weight derived from the
  sample's confidence and its residual domain gap:

  ```
  L = L_ITC + lambda * L_PACE ,   L_PACE_i = (1 - w_i) * CDCM_i + w_i * CDPA_i
  ```

Retrieval at test time uses **only the final image embeddings** (image-to-image
cosine similarity). The frozen text anchors are used only inside prompt
generation; the text targets only for the image–text loss.

## Install
```bash
conda create -n capt python=3.10 -y && conda activate capt
pip install -r requirements.txt
```

## Data (not included — download separately)
CAPT trains/evaluates on three standard benchmarks. Point `-dataset_path` at a
root arranged as below. The class splits / protocol files are already shipped
under `data/` — you only need the images.

```
<dataset_path>/
  DomainNet/<domain>/<class>/*.jpg
      domains: clipart infograph painting quickdraw real sketch
      (SnMpNet/CuMix split: 245 train + 55 val seen, 100 unseen test —
       shipped in data/DomainNet/{train,val,test}_classes.npy)
  Sketchy/  (ECCV'18 unseen split; data/Sketchy/*)
  TUBerlin/ (30-class split; data/TUBerlin/*)
```
- DomainNet: http://ai.bu.edu/M3SDA/
- Sketchy (extended) and TU-Berlin (extended): standard ZSE-SBIR releases.

## Train + evaluate (DomainNet)
One run = one held-out (unseen) domain. The source domain follows the standard
protocol: `quickdraw` for the **sketch** holdout, `sketch` for all others.
Each epoch evaluates the five protocols and prints all six metrics
({Unseen-G0, Mixed-G1, U^dCDR} x {mAP@200, Prec@200}); we report **epoch 4**
(constant learning rate). The defaults below are the paper's **CAPT-deep** recipe.

```bash
# sketch holdout
python train.py -dataset_path /path/to/datasets \
    -seen_domain quickdraw -holdout_domain sketch    -gallery_domain real -epochs 4 -batch_size 60

# clipart / painting / infograph / quickdraw holdouts
python train.py -dataset_path /path/to/datasets -seen_domain sketch -holdout_domain clipart   -gallery_domain real -epochs 4 -batch_size 60
python train.py -dataset_path /path/to/datasets -seen_domain sketch -holdout_domain painting  -gallery_domain real -epochs 4 -batch_size 60
python train.py -dataset_path /path/to/datasets -seen_domain sketch -holdout_domain infograph -gallery_domain real -epochs 4 -batch_size 60
python train.py -dataset_path /path/to/datasets -seen_domain sketch -holdout_domain quickdraw -gallery_domain real -epochs 4 -batch_size 60
```

### Cross-dataset (sketch-to-image, unseen classes)
```bash
python train.py -dataset_path /path/to/datasets -dataset Sketchy  -epochs 4 -batch_size 60
python train.py -dataset_path /path/to/datasets -dataset TUBerlin -epochs 4 -batch_size 60
```

## Expected results (DomainNet, CAPT-deep, epoch 4, mAP@200 / Prec@200)
| Holdout | Unseen (G0) | Mixed (G1) | U^dCDR |
|---|---|---|---|
| clipart   | 0.7990 / 0.7589 | 0.7436 / 0.7099 | 0.8590 / 0.5491 |
| sketch    | 0.6944 / 0.6527 | 0.6141 / 0.5841 | 0.7926 / 0.5212 |
| painting  | 0.7728 / 0.7223 | 0.7214 / 0.6778 | 0.7707 / 0.4859 |
| infograph | 0.6200 / 0.5928 | 0.5422 / 0.5230 | 0.6740 / 0.4360 |
| quickdraw | 0.3278 / 0.3017 | 0.2551 / 0.2415 | 0.3183 / 0.1278 |

Single seed (`-seed 0`), one 80GB GPU. Small run-to-run variation (~1e-3) is expected.

## Key hyperparameters (paper defaults)
| flag | default | meaning |
|---|---|---|
| `-n_prompt` | 8 | generated prompt tokens P_q |
| `-rank` | 64 | DSIP bottleneck r |
| `-gen_heads` | 4 | read-out attention heads |
| `-inject_layers` | all 12 | prompt injection depth (`0` = shallow) |
| `-text_ctx_cls` | 4 | CoOp-style class text context length |
| `-lambda_route` | 0.5 | PACE weight lambda |
| `-route_soft` | 0.1 | PACE router temperature |
| `-pace_gap` | 2 | routing signal (confidence + residual domain gap) |
| `-dsip_outhead` / `-refine_dim` | 1 / 256 | output-refinement head (~0.26M) |
| `-bank_size` | 10000 | PACE feature bank M |
| `-cdcm_tau` / `-cdcm_K` / `-cdcm_max_neg_sim` | 0.07 / 256 / 0.9 | CDCM temperature, hard negatives, false-negative cap |
| `-epochs` / `-batch_size` / `-lr` | 4 / 60 / 1e-4 | constant-LR Adam |

## Repository layout
```
clasp/   model.py (DSIP) · losses.py (PACE: CDCM/CDPA) · engine.py (train/eval) · mAP.py · metrics.py
data/    dataset loaders + shipped class-split / protocol files (no images)
train.py single entry point (train + per-epoch evaluation)
```
