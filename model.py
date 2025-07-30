import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Dict

NEG_INF = float('-1e9')    # Attention マスク用の“大きな負の無限大”はここで統一

# --- Sinusoidal Positional Encoding ---
class FixedPositionalEncoding(nn.Module):
    """学習しない正弦位置エンコーダ"""
    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build_pe(max_len))

    def _build_pe(self, length: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(length, self.d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe
  
    def forward(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        L = x.size(1)
        need = start + L
        if need > self.pe.size(0):
            extra = self._build_pe(need - self.pe.size(0)).to(self.pe.device)
            self.pe = torch.cat([self.pe, extra], dim=0)
        return x + self.pe[start : start + L]


def compute_diagonal_mask(T: int, S: int, nu: float = 0.3, device=None) -> torch.Tensor:
    # 正規化座標
    #tgt = torch.linspace(0, 1, steps=T, device=device).unsqueeze(1)  # (T,1)
    # T==1 のときはダイアゴナルマスクをゼロにして全キーを使わせる
    if T == 1:
        return torch.zeros(1, S, device=device)
    tgt = torch.arange(T, device=device).unsqueeze(1).float() / (T - 1)   
    src = torch.linspace(0, 1, steps=S, device=device).unsqueeze(0)  # (1,S)
    diff = (src - tgt).square()
    # ← ここを「正規化座標」→「生フレーム座標」に変更
    #tgt = torch.arange(T, device=device).unsqueeze(1).float()       # (T,1): [0,1,2,…,T-1]
    #src = torch.arange(S, device=device).unsqueeze(0).float()       # (1,S): [0,1,2,…,S-1]
    #diff = (src - tgt).square()                                    # 距離^2 (s - t)^2
    # ガウス重み → ログ
    mask = -diff / (2 * nu * nu)     # 直接対数空間で計算しても OK
    # 下限を大きな負数にクリップ
    neg_inf = NEG_INF
    mask = mask.clamp(min=neg_inf)

    # 完全にマスクされた行があれば、対角位置だけをゼロに戻す
    # (mask==NEG_INF) で判定
    bad_row = (mask == neg_inf).all(dim=-1)
    if bad_row.any():
        rows = bad_row.nonzero(as_tuple=False).view(-1)
        # 各 bad_row に対して、最も近い列を探して 0 に
        _, cols = (src - tgt[rows]).abs().min(dim=-1)
        mask[rows, cols] = 0.0

    return mask  # shape: (T, S)

# --- Safe Attention Mask Conversion ---
def safe_attn_mask(mask: torch.Tensor, neg_inf: float = -1e4) -> torch.Tensor:
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return mask
    mask = mask.clone()
    mask[mask != mask] = neg_inf
    mask[mask == float('-inf')] = neg_inf
    all_inf = (mask == neg_inf).all(dim=-1, keepdim=True)
    mask = mask.masked_fill(all_inf, 0.0)
    return mask

# --- Transformer Aligner for Log-Mel Spectrogram ---
class TransformerAlignerMel(nn.Module):
    def __init__(
        self,
        input_dim_mel: int = 80,
        input_dim_pitch: int = 1,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        nu: float = 0.3,
        diag_w: float = 1.0,
        ce_w: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.nu = nu
        self.diag_weight = diag_w
        self.ce_weight = ce_w

        # Projections
        self.mel_proj   = nn.Linear(input_dim_mel, d_model)
        self.pitch_proj = nn.Linear(input_dim_pitch, d_model)
        self.posenc     = FixedPositionalEncoding(d_model, max_len=8000)

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder_layers.append(nn.ModuleDict({
                'norm1':      nn.LayerNorm(d_model),
                'self_attn':  nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'norm2':      nn.LayerNorm(d_model),
                'pitch_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'norm3':      nn.LayerNorm(d_model),
                'ffn':        nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout)
                )
            }))
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_layers.append(nn.ModuleDict({
                'norm1':      nn.LayerNorm(d_model),
                'self_attn':  nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'norm2':      nn.LayerNorm(d_model),
                'pitch_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'norm3':      nn.LayerNorm(d_model),
                'cross_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'norm4':      nn.LayerNorm(d_model),
                'ffn':        nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout)
                )
            }))
            
        # Output heads and tokens
        self.out_mel          = nn.Linear(d_model, input_dim_mel)
        self.out_pitch        = nn.Linear(d_model, input_dim_pitch)
        self.token_classifier = nn.Linear(d_model, 2)
        #self.bos_token        = nn.Parameter(torch.randn(1,1,d_model))
        #self.eos_token        = nn.Parameter(torch.randn(1,1,d_model))
        self.bos_token = nn.Parameter(torch.zeros(1,1,d_model), requires_grad=False)
        self.eos_token = nn.Parameter(torch.zeros(1,1,d_model), requires_grad=False)


    def forward(
        self,
        src_mel: torch.Tensor,
        src_pitch: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_mel: torch.Tensor,
        tgt_pitch: torch.Tensor,
        tgt_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, S, _ = src_mel.size()
        _, T, _ = tgt_mel.size()
        device  = src_mel.device

        # ─── padding mask (bool) ─────────────────────────────────
        src_pad_mask = (
            torch.arange(S, device=device)
            .unsqueeze(0).expand(B, S)
            >= src_lengths.unsqueeze(1)
        )  # (B, S)  True=pad

        tgt_pad_mask = (
            torch.arange(T+1, device=device)
            .unsqueeze(0).expand(B, T+1)
            >= (tgt_lengths + 1).unsqueeze(1)
        )  # (B, T+1)

        # ─── Encoder ─────────────────────────────────────────────
        x = self.mel_proj(src_mel) + self.pitch_proj(src_pitch.unsqueeze(-1))
        x = self.posenc(x)
        p_enc = self.pitch_proj(src_pitch.unsqueeze(-1))
        for layer in self.encoder_layers:
            # Pre-Norm self-attention
            x_norm = layer['norm1'](x)
            x2, _  = layer['self_attn'](
                x_norm, x_norm, x_norm,
                key_padding_mask=src_pad_mask,
                need_weights=False
            )
            x = x + x2

            # Pre-Norm pitch-attention
            x_norm = layer['norm2'](x)
            x2p, _ = layer['pitch_attn'](
                x_norm, p_enc, p_enc,
                key_padding_mask=src_pad_mask,
                need_weights=False
            )
            x = x + x2p

            # Pre-Norm FFN
            #x = layer['ffn'](x)
            x = x + layer['ffn'](layer['norm3'](x))
            
        memory = x  # (B, S, D)

        # ──Decoder Init ─────────────────────────────────────────
        bos   = self.bos_token.expand(B, 1, self.d_model)        # (B,1,D)
        t_h   = self.mel_proj(tgt_mel)                           # (B,T,D)
        t_p   = self.pitch_proj(tgt_pitch.unsqueeze(-1))        # (B,T,D)
        x_dec = torch.cat([bos, t_h + t_p], dim=1)               # (B,T+1,D)
        x_dec = self.posenc(x_dec)

        p_dec = torch.cat(
            [torch.zeros(B,1,self.d_model,device=device), t_p],
            dim=1
        )  # (B,T+1,D)

        # 2D diagonal mask (T+1, S)
        diag = compute_diagonal_mask(T+1, S, self.nu, device)

        # build combined float mask (diag + padding), but only use if diag_weight>0
        if self.diag_weight > 0:
            neg_inf    = NEG_INF
            pad_b      = src_pad_mask.float().unsqueeze(1) * NEG_INF # (B,1,S)
            pad_b      = pad_b.expand(-1, T+1, -1)                     # (B,T+1,S)
            float_mask = (diag.unsqueeze(0) + pad_b).clamp(min=NEG_INF)
        else:
            float_mask = None

        attn_w = None
        # ──Decoder Loop ───────────────────────────────────────
        for layer in self.decoder_layers:
            # 1) causal self-attention（causal マスクを定義）
            L = x_dec.size(1)
            causal = torch.triu(
                torch.full((L, L), NEG_INF, device=device),
                diagonal=1
            )  # (L,L)
            # 先に正規化テンソルを作成してから渡す
            y_norm, x_dec = layer['norm1'](x_dec), x_dec
            y2, _ = layer['self_attn'](
                y_norm, y_norm, y_norm,
                attn_mask=causal,
                key_padding_mask=tgt_pad_mask,
                need_weights=False
            )
            y = x_dec + y2
            # 2) pitch-attn (pre-norm + post-FFN)
            y2p, _ = layer['pitch_attn'](
                layer['norm2'](y), p_dec, p_dec,
                key_padding_mask=tgt_pad_mask
            )
            y = y + y2p
            # 3) cross-attention: 2D diag + padding → 3D mask, no key_padding_mask
            if float_mask is not None:
                heads = layer['cross_attn'].num_heads
                mask3d = (
                    float_mask
                    .unsqueeze(1)                     # (B,1,T+1,S)
                    .expand(-1, heads, -1, -1)        # (B,heads,T+1,S)
                    .reshape(-1, float_mask.size(1), float_mask.size(2))  # (B*heads,T+1,S)
                )
            else:
                mask3d = None
            # capture attention weights for diagonal loss
            y2m, attn_w = layer['cross_attn'](
                layer['norm3'](y), memory, memory,
                attn_mask=mask3d,    # (B*heads,T+1,S) or None
                key_padding_mask=None,
                need_weights=True
            )
            y = y + y2m
            y = y + layer['ffn'](layer['norm4'](y))
            x_dec = y  # 次の層へ
            
        # ─── Outputs & Loss ──────────────────────────────────────
        pred_mel   = self.out_mel(x_dec)               # (B, T+1, F)
        pred_pitch = self.out_pitch(x_dec).squeeze(-1) # (B, T+1)

        # Teacher‐forcing では pred[:,1:] が tgt と対応
        # ※ pred[:,0] は BOS→初フレーム の予測なので除外
        loss_mel = F.l1_loss(pred_mel[:, 1:], tgt_mel)     # compare (B,T,F)
        loss_p   = F.l1_loss(pred_pitch[:, 1:], tgt_pitch) # compare (B,T)
        
        # diagonal regularization
        pos_s = torch.arange(S, device=device).unsqueeze(0).repeat(T+1,1)
        pos_t = torch.arange(T+1, device=device).unsqueeze(1).repeat(1,S)
        dist   = torch.abs(pos_t - pos_s).float() / S
        loss_diag = (attn_w * dist.unsqueeze(0)).sum() / (B * (T + 1))

        # ─── EOS Cross-Entropy only at final step ─────────────────
        # まずは decoder 出力から logits を取ってくる
        logits = self.token_classifier(x_dec)           # (B, T+1, 2)
        idx    = torch.arange(B, device=device)
        logits_e = logits[idx, tgt_lengths, :]         # (B, 2)
        labels_e = torch.ones(B, dtype=torch.long, device=device)
        loss_ce  = F.cross_entropy(logits_e, labels_e, label_smoothing=0.1)

        total = loss_mel + loss_p + self.diag_weight * loss_diag + self.ce_weight * loss_ce
        return total, {
            "mel_l1":   loss_mel,
            "pitch_l1": loss_p,
            "diag":     loss_diag,
            "ce":       loss_ce
        }        

    @torch.no_grad()
    def greedy_decode(
            self,
            src_mel:   torch.Tensor,    # (1, S, F)
            src_pitch: torch.Tensor,    # (1, S)
            src_len:   int,             # 有効フレーム数
            max_len:   int = 200
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        """
        - batch=1 前提
        - 学習時と同じ self/pitch/cross-attn 構成
        - dynamic window + diagonal bias を使った monotonic search
        """
        device = src_mel.device
        S = src_mel.size(1)
        heads = self.decoder_layers[0]['cross_attn'].num_heads
        NEG = NEG_INF

        # ── ① エンコーダ部 ─────────────────────────────────────
        enc = self.mel_proj(src_mel) + self.pitch_proj(src_pitch.unsqueeze(-1))
        enc = self.posenc(enc)
        p_enc = self.pitch_proj(src_pitch.unsqueeze(-1))
        for layer in self.encoder_layers:
            # self-attn
            h, _ = layer['self_attn'](
                layer['norm1'](enc), layer['norm1'](enc), layer['norm1'](enc)
            )
            enc = enc + h
            # pitch-attn
            h, _ = layer['pitch_attn'](
                layer['norm2'](enc), p_enc, p_enc
            )
            enc = enc + h
            # FFN
            enc = enc + layer['ffn'](layer['norm3'](enc))
        memory = enc                        # (1, S, D)

        # ── ② デコーダ初期化 ────────────────────────────────────
        cur         = self.posenc(self.bos_token.expand(1,1,self.d_model))  # (1,1,D)
        memory_mask = None              # dynamic window mask (heads, t, S)
        prev_peak   = 0                 # monotonic enforce 用
        outs_mel    = []
        outs_pitch  = []

        # ── ③ 生成ループ ─────────────────────────────────────────
        for step in range(max_len):
            T = cur.size(1)

            # ── (a) pitch-dec embeddings の更新 ────────────────
            if step == 0:
                p_dec = torch.zeros(1, 1, self.d_model, device=device)
            else:
                pitch_emb = self.pitch_proj(
                    torch.cat(outs_pitch, dim=1).unsqueeze(-1)  # (1, t, 1)
                )  # → (1, t, D)
                p_dec = torch.cat([
                    torch.zeros(1,1,self.d_model, device=device),
                    pitch_emb
                ], dim=1)  # (1, t+1, D)

            y = cur
            # ── (b) デコーダ層 ───────────────────────────────────
            for layer in self.decoder_layers:
                # 1) causal self-attn
                causal = torch.triu(torch.full((T, T), NEG, device=device), 1)
                h, _ = layer['self_attn'](
                    layer['norm1'](y), layer['norm1'](y), layer['norm1'](y),
                    attn_mask=causal
                )
                y = y + h

                # 2) pitch-attn
                h, _ = layer['pitch_attn'](
                    layer['norm2'](y), p_dec, p_dec
                )
                y = y + h

                # 3) cross-attn: dynamic window + diagonal bias
                #  3-1) diagonal bias
                diag2d = compute_diagonal_mask(T, S, self.nu, device)   # (T,S)
                diag3d = diag2d.unsqueeze(0).expand(heads, -1, -1)     # (heads,T,S)
                #  3-2) dynamic window merge
                if memory_mask is not None:
                    # pad to length T
                    if memory_mask.size(1) < T:
                        pad_len = T - memory_mask.size(1)
                        pad = torch.full((heads, pad_len, S), NEG, device=device)
                        memory_mask = torch.cat([memory_mask, pad], dim=1)
                    merged = diag3d + memory_mask[:, :T, :]
                else:
                    merged = diag3d
                attn_mask = safe_attn_mask(merged, neg_inf=NEG)       # (heads,T,S)

                h, att_w = layer['cross_attn'](
                    layer['norm3'](y), memory, memory,
                    attn_mask=attn_mask,
                    need_weights=True,
                    average_attn_weights=True
                )
                y = y + h

                # 4) FFN
                y = y + layer['ffn'](layer['norm4'](y))

            # ── (c) 出力プロジェクション ─────────────────────────
            last_h  = y[:, -1]                       # (1,D)
            m_raw   = self.out_mel(last_h)           # (1,F)
            p_raw   = self.out_pitch(last_h).squeeze(-1)  # (1,)

            # optional: tanh*6 で dB 範囲制限
            #m_pred  = torch.tanh(m_raw) * 6.0
            m_pred = m_raw
            p_pred  = p_raw

            outs_mel.append(m_pred.unsqueeze(1))    # list of (1,1,F)
            outs_pitch.append(p_pred.unsqueeze(1))  # list of (1,1)

            # ── (d) モノトニック窓更新 ───────────────────────────
            att_last = att_w[0, -1, :]              # (S,)
            raw_peak = int(att_last.argmax())       # 生 argmax
            peak     = max(prev_peak, raw_peak)     # monotonic enforce
            prev_peak = peak
            lo  = max(0,      peak - 5)
            hi  = min(src_len, peak + 10)
            # 新行作成 & append
            win = torch.full((heads, 1, S), NEG, device=device)
            win[:, :, lo:hi] = 0.0
            if lo == hi:
                win[:, :, peak:peak+1] = 0.0
            memory_mask = win if memory_mask is None else torch.cat([memory_mask, win], dim=1)

            # ── (e) 次トークン入力の生成 ─────────────────────────
            emb = self.mel_proj(m_pred) + self.pitch_proj(p_pred.unsqueeze(-1))
            emb = self.posenc(emb.unsqueeze(1), start=T).squeeze(1)
            cur = torch.cat([cur, emb.unsqueeze(1)], dim=1)

        # ── 最終シーケンス生成 ─────────────────────────────────
        mel_seq   = torch.cat(outs_mel,   dim=1)  # (1, max_len, F)
        pitch_seq = torch.cat(outs_pitch, dim=1)  # (1, max_len)
        return mel_seq, pitch_seq
    
    def predict(self, src_mel, src_pitch, max_len=200):
        """
        Wrapper around greedy_decode for inference.
        """
        return self.greedy_decode(src_mel, src_pitch, max_len)
