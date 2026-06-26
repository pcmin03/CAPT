"""CAPT — 메인 모델 (XDSIP + PACE, ablation 분기 제거한 클린 메인).

   train.py가 import하는 실제 메인 파일. ablation 플래그를 전부 걷어내 메인 경로만 남겼고,
   직전 ablation-포함본과 forward·파라미터수가 bit-identical(max|Δ|=0.0)임을 검증함(2026-06-16).
   ablation 재현이 필요하면 백업본 사용: CaDiP/capt_dev/clasp_ablation_backup_20260616/.
   생성기는 항상 "frozen 텍스트 앵커 + 이미지조건(instance-conditional)" 으로만 동작.

[한 줄 요약]
   얼린(frozen) CLIP ViT-B/32 위에서, "이미지마다" 비주얼 프롬프트를 생성해 12개 레이어 전부에 주입.
   프롬프트는 frozen CLIP '텍스트 클래스 앵커'를 이미지로 살짝 보정한 것에서 만들어짐(=XDSIP).
   학습 파라미터 ~1.1~1.35M 뿐(백본은 얼림). 검색은 f_prompted 끼리 코사인 유사도.

forward(image) -> {"f_prompted": (B,512) 이미지특징, "text_tune": (C,512) 클래스텍스트특징}
   둘 다 L2 정규화됨.
"""
from __future__ import annotations

from typing import List

import clip as clip_pkg
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_frozen_clip(backbone: str, device):
    # CLIP을 로드하고 모든 파라미터를 얼린다(학습 안 함). 백본은 고정, 우리는 프롬프트만 학습.
    model, _ = clip_pkg.load(backbone, device=device, jit=False)
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class CLASPNet(nn.Module):
    def __init__(self, args, class_names: List[str],
                 domain_names: List[str], device):
        super().__init__()
        self.device = device
        self.clip = _load_frozen_clip(getattr(args, "clip_backbone", "ViT-B/32"), device)
        d_v = int(self.clip.visual.conv1.out_channels)        # 비전 토큰 차원 768
        d_out = int(self.clip.text_projection.shape[1])       # 출력(공유 임베딩) 차원 512
        D_emb = self.clip.token_embedding.embedding_dim       # 텍스트 토큰 임베딩 차원 512
        self.d_v, self.d_out = d_v, d_out

        # ---- frozen 텍스트 클래스 앵커: "a photo of a {클래스}" 를 얼린 CLIP으로 인코딩 ----
        #      이 앵커들이 (1) 프롬프트 생성의 기준점, (2) ITC 손실의 타깃 역할을 함.
        with torch.no_grad():
            tok = torch.cat([clip_pkg.tokenize(f"a photo of a {c.replace('_',' ')}")
                             for c in class_names]).to(device)
            text_clip = F.normalize(self.clip.encode_text(tok).float(), dim=-1)
        self.register_buffer("text_clip", text_clip)          # (C, 512) 얼림(buffer, 학습 안 함)
        self.delta_t = nn.Parameter(torch.zeros(text_clip.shape))  # output 보정: 클래스별 텍스트 shift(0에서 시작)

        # ---- input-level 텍스트 컨텍스트 (CoOp 방식, 학습형) ----
        #      "[X][X][X][X] {클래스}" 의 X 4토큰을 학습해 텍스트 인코더 '입력'을 살짝 튜닝.
        M_cls = int(getattr(args, "text_ctx_cls", 4))
        self.n_ctx = M_cls
        self.text_ctx_cls = nn.Parameter(torch.randn(M_cls, D_emb) * 0.02)  # 0.02=CoOp 표준 초기화(작게 시작)
        placeholder = " ".join(["X"] * self.n_ctx)
        prompts = [f"{placeholder} {c.replace('_',' ')}" for c in class_names]
        with torch.no_grad():
            ctok = torch.cat([clip_pkg.tokenize(p) for p in prompts]).to(device)
        self.register_buffer("class_tokens", ctok, persistent=False)
        self.register_buffer("eot_idx", ctok.argmax(dim=-1), persistent=False)  # 각 문장의 EOT(끝) 토큰 위치

        # ---- XDSIP: 텍스트 앵커 기반 '이미지조건' 딥 프롬프트 생성기 (핵심 novelty) ----
        Pq = int(getattr(args, "n_prompt", 8))   # 생성할 비주얼 프롬프트 토큰 개수(통째로 사용)
        r = int(getattr(args, "rank", 64))       # 저차원 병목(연산량/파라미터 절약)
        n_heads = int(getattr(args, "gen_heads", 4))
        self.n_prompt, self.rank = Pq, r
        n_blocks = len(self.clip.visual.transformer.resblocks)
        # 프롬프트를 주입할 레이어: 비우면 0~11 전부(deep), "0"이면 입력층만(shallow ablation)
        layers = str(getattr(args, "inject_layers", ",".join(str(i) for i in range(n_blocks))))
        self.inject_layers = sorted({int(s) for s in layers.split(",") if s.strip() != ""
                                     if 0 <= int(s) < n_blocks})
        assert 0 in self.inject_layers, "must inject at layer 0 (input)"

        self.q_base = nn.Parameter(torch.randn(Pq, r) * 0.02)  # Pq개 프롬프트의 학습형 기본 쿼리
        self.q_proj = nn.Linear(d_v, r)                       # 현재 CLS 요약 -> 쿼리 (이미지조건 부여)
        self.kv_proj = nn.Linear(D_emb, r)                    # 텍스트 앵커 -> r차원 (Key/Value)
        self.kv_patch = nn.Linear(d_v, r)                     # 현재 패치 토큰 -> r차원
        self.xgate = nn.Parameter(torch.tensor(-3.0))         # 텍스트<-비전 보정 게이트(sigmoid(-3)~0.05, 작게 시작)
        self.attn = nn.ModuleDict()  # 레이어별 cross-attention
        self.up = nn.ModuleDict()    # 레이어별 r -> d_v 업프로젝션
        for L in self.inject_layers:
            self.attn[str(L)] = nn.MultiheadAttention(r, n_heads, batch_first=True)
            up = nn.Linear(r, d_v)
            nn.init.trunc_normal_(up.weight, std=0.02); nn.init.zeros_(up.bias)  # 0 근처에서 시작(처음엔 거의 무개입)
            self.up[str(L)] = up
        for m in (self.q_proj, self.kv_proj, self.kv_patch):
            nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

        # ---- 출력 특징 refine 헤드 (QE 역할을 DSIP 출력에 흡수; residual MLP) ----
        self.use_outhead = int(getattr(args, "dsip_outhead", 0))     # 1이면 사용(헤드라인 방법)
        if self.use_outhead:
            rdim = int(getattr(args, "refine_dim", 256))             # 병목 폭
            self.out_head = nn.Sequential(nn.Linear(d_out, rdim), nn.GELU(), nn.Linear(rdim, d_out))
            nn.init.zeros_(self.out_head[-1].weight); nn.init.zeros_(self.out_head[-1].bias)  # 0에서 시작(residual)

        # ---- CDCM용 교차도메인 특징 뱅크 (FIFO 링버퍼) : PACE 라우팅이 참조하는 프로토타입 저장소 ----
        N = int(getattr(args, "bank_size", 10000))
        self.register_buffer("cdq_features", torch.zeros(N, d_out), persistent=False)  # 저장된 특징
        self.register_buffer("cdq_classes", torch.full((N,), -1, dtype=torch.long), persistent=False)  # 클래스 라벨
        self.register_buffer("cdq_domains", torch.full((N,), -1, dtype=torch.long), persistent=False)  # 도메인 라벨
        self.register_buffer("cdq_valid", torch.zeros(N, dtype=torch.bool), persistent=False)  # 채워졌는지 표시
        self.register_buffer("cdq_ptr", torch.zeros(1, dtype=torch.long), persistent=False)  # 다음 쓸 위치

        self.float()  # 생성기/텍스트컨텍스트는 fp32 (CLIP은 이미 float)
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[CLASP] XDSIP inject_layers={self.inject_layers} Pq={Pq} rank={r}; "
              f"trainable={n_train:,} (frozen CLIP {getattr(args,'clip_backbone','ViT-B/32')})")

    # ------------------------------------------------------------------
    def _gen_prompt(self, L: int, cls_summary: torch.Tensor, patches: torch.Tensor):
        """Language-anchored prompt read-out (단일 cross-attention).

        비주얼 프롬프트를 frozen CLIP '텍스트-클래스 앵커'에서 *읽어낸다*:
        학습 쿼리가 앵커에 cross-attention해 Pq개 프롬프트 토큰을 얻는다.
        앵커는 미리 현재 패치로 가볍게 image-conditioning되어(게이트), 프롬프트가
        instance-aware하면서도 language manifold 위에 머문다. (step 분리 아님 — 한 번의 read-out)
        """
        B = cls_summary.shape[0]
        scale = self.kv_proj.out_features ** -0.5
        # frozen language anchors -> r차원 key/value
        anchor = self.kv_proj(self.text_clip.float()).unsqueeze(0).expand(B, -1, -1)    # (B,C,r)
        patch = self.kv_patch(patches.float())                                          # (B,P,r)
        # 앵커를 패치로 가볍게 image-conditioning (게이트된 잔차) — 각 앵커가 매칭 패치 쪽으로 살짝 이동
        img_cond = torch.bmm(torch.softmax(anchor @ patch.transpose(1, 2) * scale, dim=-1), patch)
        anchor = anchor + torch.sigmoid(self.xgate) * img_cond                          # (B,C,r)
        # static 학습 쿼리가 앵커에서 프롬프트를 읽어냄 (논문 default: instance_cond=0).
        q = self.q_base.unsqueeze(0)
        if int(getattr(self, "instance_cond", 0)):                                       # optional per-image query (논문서 미채택)
            query = q + self.q_proj(cls_summary.float()).unsqueeze(1)                     # (B,Pq,r)
        else:
            query = q.expand(cls_summary.shape[0], -1, -1)                               # (B,Pq,r) static
        prompt, _ = self.attn[str(L)](query, anchor, anchor)                            # (B,Pq,r)
        return self.up[str(L)](prompt)                                                  # (B,Pq,d_v)

    def _forward_visual(self, image: torch.Tensor):
        # 얼린 CLIP 비전 인코더를 돌리되, 지정 레이어마다 생성 프롬프트를 끼워넣는다.
        v = self.clip.visual
        x = v.conv1(image.type(self.clip.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)        # (B,P,d_v) 패치 토큰들
        B = x.shape[0]
        cls = v.class_embedding.to(x.dtype) + torch.zeros(
            B, 1, x.shape[-1], dtype=x.dtype, device=x.device)           # CLS 토큰
        cls = cls + v.positional_embedding[0:1, :].to(x.dtype)            # 위치 임베딩(CLS)
        patches = x + v.positional_embedding[1:, :].to(x.dtype)          # 위치 임베딩(패치)
        Pq = self.n_prompt
        # 입력층(layer 0) 프롬프트 생성 후 [CLS, 프롬프트, 패치] 로 이어붙임
        P0 = self._gen_prompt(0, cls.squeeze(1), patches).to(patches.dtype)
        seq = v.ln_pre(torch.cat([cls, P0, patches], dim=1)).permute(1, 0, 2)  # LND 순서
        blocks = list(v.transformer.resblocks)
        for i, blk in enumerate(blocks):
            seq = blk(seq)                                               # 한 트랜스포머 블록 통과
            nl = i + 1
            if nl in self.inject_layers and nl < len(blocks):
                # 다음 레이어용 프롬프트를 '현재' CLS/패치로 새로 생성해 교체(deep, 매 레이어 갱신)
                P_L = self._gen_prompt(nl, seq[0], seq[1 + Pq:].permute(1, 0, 2)).to(seq.dtype)
                seq = torch.cat([seq[0:1], P_L.permute(1, 0, 2), seq[1 + Pq:]], dim=0)
        cls_f = (v.ln_post(seq[0]) @ v.proj).float()                     # CLS 토큰만 공유공간(512)으로 투영
        return cls_f / cls_f.norm(dim=-1, keepdim=True)                  # L2 정규화 -> 이미지 특징

    def _encode_text(self) -> torch.Tensor:
        # 텍스트 인코딩: "[X..X] {클래스}"에서 X자리(BOS 다음)를 학습형 text_ctx_cls로 바꿔 얼린 인코더 통과.
        dt = self.clip.dtype
        x = self.clip.token_embedding(self.class_tokens).type(dt)
        C = x.shape[0]
        ctx = self.text_ctx_cls
        ctx = ctx.unsqueeze(0).expand(C, -1, -1).type(dt)
        x = torch.cat([x[:, :1], ctx, x[:, 1 + self.n_ctx:]], dim=1)     # [BOS, 학습컨텍스트, 클래스토큰...]
        x = x + self.clip.positional_embedding.type(dt)
        x = self.clip.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)   # 얼린 텍스트 트랜스포머
        x = self.clip.ln_final(x).type(dt)
        x = x[torch.arange(C, device=x.device), self.eot_idx] @ self.clip.text_projection  # EOT 토큰을 클래스 특징으로
        return F.normalize(x.float(), dim=-1)

    def forward(self, image: torch.Tensor) -> dict:
        # 메인 forward: 이미지특징(프롬프트 적용) + 클래스텍스트특징을 반환.
        f = self._forward_visual(image)                                 # 프롬프트 적용된 이미지 특징
        if self.use_outhead:                                            # refine 헤드(있으면 residual로 다듬기)
            f = F.normalize(f + self.out_head(f), dim=-1)
        text_tune = F.normalize(self._encode_text() + self.delta_t, dim=-1)  # 텍스트 = 입력컨텍스트 + 출력shift
        return {"f_prompted": f, "text_tune": text_tune}

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor, classes: torch.Tensor, domains: torch.Tensor):
        # 배치 특징을 뱅크(링버퍼)에 정규화해서 넣음. 가장 오래된 자리를 덮어씀(FIFO).
        feats = F.normalize(feats, dim=-1).float()
        B, N = feats.shape[0], self.cdq_features.shape[0]
        idx = (torch.arange(B, device=feats.device) + int(self.cdq_ptr.item())) % N
        self.cdq_features[idx] = feats
        self.cdq_classes[idx] = classes.to(self.cdq_classes.dtype)
        self.cdq_domains[idx] = domains.to(self.cdq_domains.dtype)
        self.cdq_valid[idx] = True
        self.cdq_ptr[0] = (int(self.cdq_ptr.item()) + B) % N

    def bank(self):
        # 현재 채워진(valid) 뱅크 항목만 반환 (특징, 클래스, 도메인). 손실(CDCM/PACE)에서 사용.
        v = self.cdq_valid
        return self.cdq_features[v], self.cdq_classes[v], self.cdq_domains[v]
