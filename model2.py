"""
Full Mel-Pitch Aligner with static diagonal mask, EOS stopping,
matching original constructor signature.
"""

import math
from typing import Tuple, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9  # large negative for masking

# ──────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        L = x.size(1)
        if start + L > self.pe.size(0):
            extra = torch.zeros(start+L-self.pe.size(0), self.pe.size(1), device=x.device)
            self.pe = torch.cat([self.pe, extra], dim=0)
        return x + self.pe[start:start+L]

# ──────────────────────────────────────────────────────────
# Static Diagonal Gaussian Mask
# ──────────────────────────────────────────────────────────
def diag_mask(
    tgt_len: int,
    src_len: int,
    nu: float = 0.3,
    alpha: float = 0.3,
    device=None
) -> torch.Tensor:
    """
    tgt_len の各ステップと src_len の各位置の「近さ」を
    ガウス型 m = −(s−t)^2/(2ν^2) で表したあと、
    |s−t| > alpha の要素は完全マスク（NEG_INF）にする
    """
    if tgt_len == 1:
        return torch.zeros(1, src_len, device=device)

    # 正規化時間軸 t ∈ [0,1], s ∈ [0,1]
    t = torch.arange(tgt_len, device=device).unsqueeze(1) / (tgt_len - 1)  # [T,1]
    s = torch.linspace(0, 1, src_len, device=device).unsqueeze(0)         # [1,S]

    # ガウス状スコア
    m = -(s - t).pow(2) / (2 * nu * nu)  # [T,S]

    # 閾値を超えた部分は完全マスク
    diff = (s - t).abs()
    mask = torch.where(
        diff <= alpha,
        m,
        torch.full_like(m, NEG_INF)
    )
    return mask
'''
def diag_mask(tgt_len: int, src_len: int, nu: float = 0.3, device=None) -> torch.Tensor:
    if tgt_len == 1:
        return torch.zeros(1, src_len, device=device)
    t = torch.arange(tgt_len, device=device).unsqueeze(1) / (tgt_len - 1)
    s = torch.linspace(0, 1, src_len, device=device).unsqueeze(0)
    m = -(s - t).pow(2) / (2 * nu * nu)
    return m.clamp(min=NEG_INF)
'''

# ──────────────────────────────────────────────────────────
# Custom Encoder Layer with Pitch-Attention
# ──────────────────────────────────────────────────────────
class EncoderLayerWithPitch(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pitch_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor, p_enc: torch.Tensor, key_padding_mask=None):
        h,_ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                             key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        h,_ = self.pitch_attn(self.norm2(x), p_enc, p_enc,
                              key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        x = x + self.ffn(self.norm3(x))
        return x

# ──────────────────────────────────────────────────────────
# Custom Decoder Layer with Pitch-Attention
# ──────────────────────────────────────────────────────────
class DecoderLayerAlign(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pitch_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
    def forward(self, y: torch.Tensor, p_dec: torch.Tensor,
                memory: torch.Tensor, tgt_mask=None, memory_mask=None):
        h,_ = self.self_attn(self.norm1(y), self.norm1(y), self.norm1(y),
                             attn_mask=tgt_mask, need_weights=False)
        y = y + h
        h,_ = self.pitch_attn(self.norm2(y), p_dec, p_dec, need_weights=False)
        y = y + h
        h, att_w = self.cross_attn(self.norm3(y), memory, memory,
                                   attn_mask=memory_mask, need_weights=True)
        y = y + h
        y = y + self.ffn(self.norm4(y))
        return y, att_w

# ──────────────────────────────────────────────────────────
# Full Model Matching Original Signature
# ──────────────────────────────────────────────────────────
import math
from typing import Tuple, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9  # large negative for masking

# ... (PositionalEncoding, diag_mask, EncoderLayerWithPitch, DecoderLayerAlign unchanged) ...

class MelPitchAligner(nn.Module):
    def __init__(self,
                 input_dim_mel:      int = 80,
                 input_dim_pitch:    int = 1,
                 d_model:            int = 256,
                 nhead:              int = 4,
                 num_layers:         int = 3,
                 dim_feedforward:    int = 512,
                 dropout:            float = 0.1,
                 nu:                 float = 0.3,
                 diag_w:             float = 1.0,
                 ce_w:               float = 1.0,
                 free_run_steps:     int = 10,
                 free_run_w:         float = 0.1,
                 use_f0:             bool = True):
        super().__init__()
        self.use_f0 = use_f0
        self.free_run_steps = free_run_steps
        self.free_run_w = free_run_w
        self.nu      = nu
        self.diag_w  = diag_w
        self.ce_w    = ce_w
        self.d_model = d_model
        
        # embeddings
        self.mel_proj   = nn.Linear(input_dim_mel, d_model)
        # 常に「1→d_model」の線形層とし、無効時は重みをゼロ初期化
        self.pitch_proj = nn.Linear(input_dim_pitch, d_model)
        if not use_f0:
            nn.init.constant_(self.pitch_proj.weight, 0.0)
            nn.init.constant_(self.pitch_proj.bias,   0.0)
        self.out_f0 = nn.Linear(d_model, input_dim_pitch)
        if not use_f0:
            nn.init.constant_(self.out_f0.weight, 0.0)
            nn.init.constant_(self.out_f0.bias,   0.0)        
        self.pos_enc    = PositionalEncoding(d_model)

        # stacks
        self.encoder_layers = nn.ModuleList([
            EncoderLayerWithPitch(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayerAlign(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # heads
        self.out_mel   = nn.Linear(d_model, input_dim_mel)
        self.token_cls = nn.Linear(d_model, 2)

        # BOS token
        self.register_buffer("bos", torch.zeros(1,1,d_model))

    def encode(self,
               src_mel: torch.Tensor,
               src_f0:  torch.Tensor) -> torch.Tensor:
        """
        Encoder part only, returns detached memory for free-run decoding.
        """
        device = src_mel.device
        if self.use_f0:
            p_enc = self.pitch_proj(src_f0.unsqueeze(-1))
        else:
            # f0 無効時はゼロ埋め
            p_enc = torch.zeros(src_mel.size(0), src_mel.size(1), self.d_model, device=device)
        enc = self.pos_enc(self.mel_proj(src_mel) + p_enc)

        for layer in self.encoder_layers:
            enc = layer(enc, p_enc)
        return enc.detach()

    def forward(self,
                src_mel:  torch.Tensor,
                src_f0:   torch.Tensor,
                src_lens: torch.Tensor,
                tgt_mel:  torch.Tensor,
                tgt_f0:   torch.Tensor,
                tgt_lens: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, S, _ = src_mel.shape
        T       = tgt_mel.size(1)
        device  = src_mel.device

        # teacher-forcing decode + loss
        # encode
        src = self.mel_proj(src_mel) + self.pitch_proj(src_f0.unsqueeze(-1))
        src = self.pos_enc(src)
        p_enc = self.pitch_proj(src_f0.unsqueeze(-1))
        for layer in self.encoder_layers:
            src = layer(src, p_enc)
        memory = src

        # prepare decoder inputs
        bos = self.bos.expand(B, -1, -1).to(device)
        tgt_emb = self.mel_proj(tgt_mel) + self.pitch_proj(tgt_f0.unsqueeze(-1))
        y = self.pos_enc(torch.cat([bos, tgt_emb], dim=1))
        p_dec_full = torch.cat([torch.zeros_like(bos), self.pitch_proj(tgt_f0.unsqueeze(-1))], dim=1)

        # masks
        causal = torch.triu(torch.full((T+1, T+1), NEG_INF, device=device), diagonal=1)
        diag   = diag_mask(T+1, S, self.nu, device=device)

        total_diag = 0.0
        for layer in self.decoder_layers:
            y, attw = layer(y, p_dec_full, memory, tgt_mask=causal, memory_mask=diag)
            total_diag += (attw * (torch.abs(
                torch.arange(T+1, device=device).unsqueeze(1).float()
                - torch.arange(S, device=device).unsqueeze(0).float()
            )/S)).sum()/(B*(T+1))

        pred_mel = self.out_mel(y)[:,1:]
        loss_mel = F.l1_loss(pred_mel, tgt_mel)
        total    = loss_mel + self.diag_w * total_diag
        metrics  = {"mel_l1": loss_mel.detach(), "diag": total_diag.detach()}
        if self.use_f0:
            pred_f0 = self.out_f0(y).squeeze(-1)[:,1:]
            loss_f0 = F.l1_loss(pred_f0, tgt_f0)
            total  += loss_f0
            metrics["f0_l1"] = loss_f0.detach()        

        # free-run loss: batch 全体で部分的に自己回帰デコード
        if self.training and self.free_run_w > 0.0:
            # バッチ内の最小ターゲット長を取得
            min_len = tgt_lens.min().item()
            # L は free_run_steps と min_len の小さいほう
            L       = min(self.free_run_steps, min_len)
            if L > 0:
                # 安全な開始位置 p をランダム選択
                p = torch.randint(0, min_len - L + 1, (1,)).item()
                # memory はすでに detach 済みなのでそのまま使う
                memory_detach = memory.detach()
                # バッチ一括で free-run
                mel_free, f0_free = self.decode_free_run(memory_detach, p, L)
                # 教師信号を対応する位置から切り出し
                mel_gt = tgt_mel[:, p:p+L]
                # バッチ全体の L1 損失
                loss_free_m = F.l1_loss(mel_free, mel_gt)
                total      += self.free_run_w * loss_free_m
                metrics["free_mel"] = loss_free_m.detach()
                if self.use_f0:
                    f0_gt       = tgt_f0[:, p:p+L]
                    loss_free_f = F.l1_loss(f0_free, f0_gt)
                    total      += self.free_run_w * loss_free_f
                    metrics["free_f0"] = loss_free_f.detach()        
        '''
        if self.training:
            idx = torch.randint(0, B, (1,)).item()
            memory_i = self.encode(src_mel[idx:idx+1], src_f0[idx:idx+1])
            # ランダムに開始位置 p
            L = min(self.free_run_steps, T)
            p = torch.randint(0, T - L + 1, (1,)).item()
            # signature changed to (memory, start, max_len)
            mel_free, f0_free = self.decode_free_run(memory_i, p, L)            
            # 教師信号も p:p+L だけ抜き出して比較
            mel_gt = tgt_mel[idx:idx+1, p:p+L]
            loss_free_m = F.l1_loss(mel_free, mel_gt)
            if self.use_f0:
                f0_gt  = tgt_f0[idx:idx+1, p:p+L]
                loss_free_f = F.l1_loss(f0_free, f0_gt)
                total = total + self.free_run_w * (loss_free_m + loss_free_f)
                metrics.update({"free_mel": loss_free_m.detach(), "free_f0": loss_free_f.detach()})
            else:
                total = total + self.free_run_w * loss_free_m
                metrics.update({"free_mel": loss_free_m.detach()})
        '''
        
        return total, metrics

    @torch.no_grad()
    def greedy_decode(self,
                      src_mel:  torch.Tensor,
                      src_f0:   torch.Tensor,
                      src_len:  int,
                      max_len:  int = 200
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        # unchanged 推論用メソッド
        assert src_mel.size(0) == 1, f"Batch size must be 1, got {src_mel.size(0)}"
        self.eval()
        device = src_mel.device
        S      = src_mel.size(1)
        if self.use_f0:
            p_enc = self.pitch_proj(src_f0.unsqueeze(-1))
        else:
            p_enc = torch.zeros(1, S, self.d_model, device=device)
        enc = self.pos_enc(self.mel_proj(src_mel) + p_enc)                               
        for layer in self.encoder_layers:
            enc = layer(enc, p_enc)
        memory = enc
        full_causal = torch.triu(torch.full((max_len, max_len), NEG_INF, device=device), diagonal=1)
        cur = self.pos_enc(self.bos.clone().to(device))
        outs_m, outs_f = [], []
        for step in range(max_len):
            T = cur.size(1)
            causal = full_causal[:T, :T]
            if step == 0:
                p_dec = torch.zeros_like(cur)
            else:
                f_stack = torch.cat(outs_f, dim=1)
                p_dec    = torch.cat([torch.zeros_like(cur[:,:1]), self.pitch_proj(f_stack)], dim=1)
            p_dec = p_dec.detach()
            diag = diag_mask(T, S, self.nu, device=device)
            y = cur
            for layer in self.decoder_layers:
                y, _ = layer(y, p_dec, memory, tgt_mask=causal, memory_mask=diag)
            last = y[:, -1]
            m_raw = self.out_mel(last)
            f_raw = self.out_f0(last)
            outs_m.append(m_raw.unsqueeze(1))
            outs_f.append(f_raw.unsqueeze(1))
            m_emb = self.mel_proj(m_raw).unsqueeze(1)
            if self.use_f0:
                p_emb = self.pitch_proj(f_raw.unsqueeze(-1))
            else:
                p_emb = torch.zeros_like(m_emb)                               
            emb   = self.pos_enc(m_emb + p_emb, start=T)
            cur = torch.cat([cur, emb], dim=1)
            cur = cur.detach()
        mel_seq = torch.cat(outs_m, dim=1)
        # f₀もゼロ列で返す
        f0_seq  = torch.cat(outs_f, dim=1) if self.use_f0 else torch.zeros_like(mel_seq[..., :1]).squeeze(-1)
        return mel_seq, f0_seq

    def decode_free_run(self,
                        memory: torch.Tensor,
                        start: int,
                        max_len: int
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        学習時用自己回帰デコード: encode() の出力 memory を再利用
        グラフを逐次切り離して OOM を防ぎます。
        """
        device = memory.device
        S      = memory.size(1)
        # memory は事前に detach してエンコーダのグラフを切り離す
        memory = memory.detach()

        # 全長 causal mask
        full_causal = torch.triu(
            torch.full((max_len, max_len), NEG_INF, device=device),
            diagonal=1
        )        

        # autoregressive loop
        cur = self.pos_enc(self.bos.expand(memory.size(0), -1, -1).to(device), start=start)
        
        outs_m, outs_f = [], []
        for step in range(max_len):
            T = cur.size(1)
            causal = full_causal[:T, :T]
            if step == 0 or not self.use_f0:
                p_dec = torch.zeros_like(cur)
            else:
                f_stack = torch.cat(outs_f, dim=1)
                p_dec    = torch.cat([
                    torch.zeros_like(cur[:, :1]),
                    self.pitch_proj(f_stack)
                ], dim=1)
            diag = diag_mask(T, S, self.nu, device=device)

            y = cur
            for layer in self.decoder_layers:
                y, _ = layer(
                    y, p_dec, memory,
                    tgt_mask=causal,
                    memory_mask=diag
                )

            last  = y[:, -1]                                      # [B, d_model]
            m_raw = self.out_mel(last)                            # [B, mel_dim]
            f_raw = self.out_f0(last) if self.use_f0 else torch.zeros(memory.size(0), 1, device=device)
            outs_m.append(m_raw.unsqueeze(1))                     # [B,1,mel_dim]
            outs_f.append(f_raw.unsqueeze(1))                     # [B,1,1]

            # 次ステップ入力 embedding (位置は start+step から)
            m_emb = self.mel_proj(m_raw).unsqueeze(1)
            p_emb = self.pitch_proj(f_raw.unsqueeze(-1))
            emb   = self.pos_enc(m_emb + p_emb, start=start+step)

            cur = torch.cat([cur, emb], dim=1)
            # detach() で古いグラフを解放
            cur = cur.detach()

        mel_seq = torch.cat(outs_m, dim=1)                        # [B, max_len, mel_dim]
        f0_seq  = torch.cat(outs_f, dim=1).squeeze(-1) if self.use_f0 else torch.zeros(memory.size(0), max_len, device=device)
        return mel_seq, f0_seq
