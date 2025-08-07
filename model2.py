"""
Full Mel-Pitch Aligner with static diagonal mask, EOS stopping,
matching original constructor signature.
"""

import math
from typing import Tuple, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from prepost import PreNet, PostNet

NEG_INF = -1e9  # large negative for masking
   
def compute_diagonal_weights(B, T, S, src_lengths, tgt_lengths, nu=0.3, batch_first=True):
    with torch.no_grad():
        weights = torch.ones(B, T, S, device=src_lengths.device)
        for b in range(B):
            # build normalized targets and sources
            tgt = torch.arange(tgt_lengths[b], device=src_lengths.device).reshape(-1, 1).repeat(1, src_lengths[b]).float()
            tgt = tgt / tgt_lengths[b]
            src = torch.arange(src_lengths[b], device=src_lengths.device).reshape(1, -1).repeat(tgt_lengths[b], 1).float()
            src = src / src_lengths[b]
            #weight = torch.exp(- (src - tgt).pow(2) / (2.0 * nu * nu))
            #weights[b, :tgt_lengths[b], :src_lengths[b]] -= weight
            gauss = torch.exp(- (src - tgt).pow(2) / (2.0 * nu * nu))
            weights[b, :tgt_lengths[b], :src_lengths[b]] = gauss
    if not batch_first:
        weights = weights.transpose(1, 0, 2)
    return weights

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

# ──────────────────────────────────────────────────────────
# Custom Encoder Layer with Pitch-Attention
# ──────────────────────────────────────────────────────────
class EncoderLayerWithPitch(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout, use_f0=False):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pitch_attn = None
        if use_f0:
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
        # pitch-attention: use_f0=False のときはスキップ
        if p_enc is not None:
            h,_ = self.pitch_attn(
                self.norm2(x), p_enc, p_enc,
                key_padding_mask=key_padding_mask, need_weights=False
            )
            x = x + h        
        x = x + self.ffn(self.norm3(x))
        return x

# ──────────────────────────────────────────────────────────
# Custom Decoder Layer with Pitch-Attention
# ──────────────────────────────────────────────────────────
class DecoderLayerAlign(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout, use_f0=False):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # forward 時の attention weights をキャプチャする
        self.last_attn: torch.Tensor = None
        self.pitch_attn=None
        if use_f0:
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
        # 4. Always regenerate tgt_mask (no fallback)
        T = y.size(1)
        tgt_mask = torch.triu(
            torch.full((T, T), NEG_INF, device=y.device),
            diagonal=1
        )
        tgt_mask.fill_diagonal_(0.0)        
        # self-attention (we don’t need its weights for diagonal loss)
        h, _ = self.self_attn(
            self.norm1(y), self.norm1(y), self.norm1(y),
            attn_mask=tgt_mask, need_weights=False
        )
        y = y + h
        # pitch-attention: use_f0=False のときはスキップ
        if p_dec is not None:
            h, _ = self.pitch_attn(
                self.norm2(y), p_dec, p_dec,
                need_weights=False
            )
            y = y + h
        B = y.size(0)
        if memory_mask is not None and memory_mask.dim() == 3:
            memory_mask = memory_mask.repeat(B, 1, 1)

        # --- 合体マスク = soft bias + hard mask ---
        # ❶ soft diagonal bias を計算（T×S）
        T_dec = y.size(1)
        S_enc = memory.size(1)
        t_norm = torch.arange(T_dec, device=y.device).float().unsqueeze(1) / max(T_dec - 1, 1)
        s_norm = torch.arange(S_enc, device=y.device).float().unsqueeze(0) / max(S_enc - 1, 1)
        diag_bias = -torch.abs(t_norm - s_norm) * 5.0   # scale factor は任意

        bias = diag_bias            
        if memory_mask is not None:             # (H,T,S) or (T,S)
            if memory_mask.dim() == 3:          # broadcast (H,T,S) → (T,S)
                memory_mask = memory_mask.mean(0)
            bias = bias + memory_mask           # -∞ を重ねる

        h, att_w = self.cross_attn(
            self.norm3(y), memory, memory,
            attn_mask=bias,
            need_weights=True, average_attn_weights=False
        )

        # 保存：最新の cross-attention weight (B, num_heads, T_tgt, T_src)
        self.last_attn = att_w
        y = y + h
        # feed-forward
        y = y + self.ffn(self.norm4(y))
        # 戻り値：出力テンソルと保存済み attention weights
        return y, self.last_attn

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
    def __init__(
        self,
        input_dim_mel:   int   = 80,
        input_dim_pitch: int   = 1,
        d_model:         int   = 256,
        nhead:           int   = 4,
        num_layers:      int   = 3,
        dim_feedforward: int   = 512,
        dropout:         float = 0.1,
        nu:              float = 0.3,
        diag_w:         float = 1.0,
        ce_w:            float = 1.0,
        ga_w:           float = 2.0,
        use_f0:          bool  = True,
        mono_w:         float = 0.1
    ):
        super().__init__()
        self.nhead    = nhead
        self.use_f0   = use_f0
        self.nu       = nu
        self.diag_w   = diag_w
        self.ce_w     = ce_w
        self.ga_w     = ga_w
        self.d_model  = d_model
        self.mono_w   = mono_w

        # embeddings + PreNet
        self.prenet      = PreNet(mel_dim=input_dim_mel, model_dim=d_model, dropout=dropout)
        
        self.pitch_proj  = nn.Linear(input_dim_pitch, d_model)
        if not use_f0:
            nn.init.constant_(self.pitch_proj.weight, 0.0)
            nn.init.constant_(self.pitch_proj.bias,   0.0)
        self.out_f0      = nn.Linear(d_model, input_dim_pitch)
        if not use_f0:
            nn.init.constant_(self.out_f0.weight, 0.0)
            nn.init.constant_(self.out_f0.bias,   0.0)

        # positional encoding
        self.pos_enc = PositionalEncoding(d_model)

        # encoder / decoder stacks
        self.encoder_layers = nn.ModuleList([
            EncoderLayerWithPitch(d_model, nhead, dim_feedforward, dropout, self.use_f0)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayerAlign(d_model, nhead, dim_feedforward, dropout, self.use_f0)
            for _ in range(num_layers)
        ])

        # output heads
        self.out_mel   = nn.Linear(d_model, input_dim_mel)
        self.token_cls = nn.Linear(d_model, 2)
        self.postnet   = PostNet(model_dim=d_model, mel_dim=input_dim_mel, dropout=dropout)        

        # BOS token
        self.bos = nn.Parameter(torch.randn(1, 1, d_model))  # trainable BOS

    def generate_square_subsequent_mask(self, seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones((seq_len, seq_len))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, NEG_INF).masked_fill(mask == 1, 0.0)
        return mask

    def generate_masks(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_len: torch.Tensor,
        tgt_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = src.shape
        T       = tgt.shape[1]
        src_mask = torch.zeros((S, S), dtype=torch.bool, device=src.device)
        tgt_mask = self.generate_square_subsequent_mask(T).to(src.device)

        src_pad_mask = torch.ones((B, S), dtype=torch.bool, device=src.device)
        tgt_pad_mask = torch.ones((B, T), dtype=torch.bool, device=src.device)
        for b in range(B):
            src_pad_mask[b, :src_len[b]] = False
            tgt_pad_mask[b, :tgt_len[b]] = False

        return src_mask, tgt_mask, src_pad_mask, tgt_pad_mask

    def get_attention_weight(self):
        # assumes DecoderLayerAlign stores last attention weights
        weights = [layer.last_attn for layer in self.decoder_layers]
        return weights

    def forward(
        self,
        src_mel:   torch.Tensor,
        tgt_mel:   torch.Tensor,
        src_f0:    torch.Tensor,
        tgt_f0:    torch.Tensor,
        src_len:   torch.Tensor,
        tgt_len:   torch.Tensor,
        nu:        float = None,
        weight:    float = None,
        *,
        epoch:     Optional[int] = None,
        batch_idx: Optional[int] = None,
        save_dir:  str = None,
        keep_maps: bool = False    
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # prepare sequences
        B, S, D = src_mel.size()
        zeros = torch.zeros((B,1,D), device=src_mel.device)
        tgtz = torch.cat((zeros, tgt_mel), dim=1)
        tgt_out = tgtz[:,1:,:]
        tgt_in  = tgtz[:,:-1,:]

        # pitch alignment for tgt
        zeros_f0 = torch.zeros((B,1), device=src_f0.device)
        f0z = torch.cat((zeros_f0, tgt_f0), dim=1)
        f0_in = f0z[:,:-1]

        # embeddings with optional F0 conditioning
        mel_enc = self.prenet(src_mel)
        if self.use_f0:
            p_src = self.pitch_proj(src_f0.unsqueeze(-1))
            y = self.pos_enc(mel_enc + p_src)
        else:
            y = self.pos_enc(mel_enc)
        mel_dec = self.prenet(tgt_in)
        if self.use_f0:
            p_tgt = self.pitch_proj(f0_in.unsqueeze(-1))
            z = self.pos_enc(mel_dec + p_tgt)
        else:
            z = self.pos_enc(mel_dec)

        # masks
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = \
            self.generate_masks(y, z, src_len, tgt_len)

        # prepare hard diagonal mask for cross-attention
        # assume batch size B, but simplify for B=1 or loop per b if B>1
        hard = diag_mask(tgt_len[0].item(), src_len[0].item(),
                         nu=self.nu, alpha=0.3, device=y.device)  # (T, S)
        hard = hard.unsqueeze(0).repeat(self.nhead, 1, 1)      # (H, T, S)

        # encoder (with optional pitch conditioning)
        for layer in self.encoder_layers:
            p_enc = p_src if self.use_f0 else None
            y = layer(
                y,
                p_enc,
                key_padding_mask=src_pad_mask
            )        
        memory = y

        # ── デコーダ入力の準備 ──
        if self.use_f0:
            p_dec = torch.cat([
                torch.zeros((B,1,self.d_model), device=src_mel.device),
                self.pitch_proj(f0_in.unsqueeze(-1))
            ], dim=1)
        else:
            p_dec = None

        # ❶ hard mask は p_dec 定義後に作成する
        T_max = z.size(1)       # 現在のデコーダ長
        S_max = memory.size(1)  # エンコーダ出力長
        hard = diag_mask(
            tgt_len=T_max,
            src_len=S_max,
            nu=self.nu,
            alpha=0.3,
            device=y.device
        )  # (T_max, S_max)

        # デコーダループ：memory_mask に hard mask を渡す
        for layer in self.decoder_layers:
            z, _ = layer(
                z,
                p_dec,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=hard
            )

        # postnet
        y = self.postnet(z)

        # losses
        mloss = F.l1_loss(y, tgt_out)
        nu = nu or self.nu
        weight = weight or self.ce_w
        # compute diagonal attention weights
        diag_weights = compute_diagonal_weights(
            B, tgt_out.shape[1], S, src_len, tgt_len, nu
        )
        # デコーダレイヤーから attention 重みを取得し、理想の対角重みと合わせて損失を計算
        attn_weights = self.get_attention_weight()
        diag_loss = self._diagonal_attention_loss(
            attn_weights,
            diag_weights,
            src_len,
            tgt_len
        )
        # ───────── Guided-attention loss (Gaussian) ───────────
        sigma = 0.2
        # 1. Use only top decoder layer for guided loss
        attn_weights = self.get_attention_weight()
        att_last     = attn_weights[-1]                     # (B, H, T, S)
        att_mean     = att_last.mean(dim=1)                 # (B, T, S)

        # 2 & 3. Increase sigma & implement warm-up phase
        # ---- 各損失の重み決定（ウォームアップ考慮） ----
        sigma = 0.4
        if epoch is not None and epoch < 10:
            diag_w_cur = self.diag_w      # diagonal attention warm-up
            ce_w_cur   = self.ce_w
            ga_w_cur   = self.ga_w
        else:
            diag_w_cur = self.diag_w
            ce_w_cur   = self.ce_w
            ga_w_cur   = self.ga_w        
        ga_loss  = 0.0
        for b in range(B):
            t_len = tgt_len[b].item()
            s_len = src_len[b].item()
            # 現バッチの有効領域を切り出し
            att_sub = att_mean[b, :t_len, :s_len]          # (t,s)
            # 正規化座標
            t_norm = torch.arange(t_len,
                                  device=att_sub.device).float().unsqueeze(1) / max(t_len-1, 1)
            s_norm = torch.arange(s_len,
                                  device=att_sub.device).float().unsqueeze(0) / max(s_len-1, 1)
            ga_mask = torch.exp(- (t_norm - s_norm).pow(2) / (2 * sigma * sigma))  # (t,s)
            ga_loss += torch.mean(torch.abs(att_sub - ga_mask))
        ga_loss = ga_loss / B
        
        # ─── total ────────────────────────────────────────────
        loss = mloss + (diag_w_cur * 5.0) * diag_loss + ga_w_cur * ga_loss
        

        # ──────────────────────────────────────────────────────
        #  □  可視化: epoch > 20 のとき最上位層のアテンションを保存
        # ──────────────────────────────────────────────────────
        if (
            epoch is not None
            and batch_idx is not None
            and epoch > 20
            and attn_weights  # 念のため空チェック
            and save_dir is not None
        ):
            try:
                import matplotlib.pyplot as plt
                from pathlib import Path

                # 最上位層（最後のデコーダ層）を取得: (B, H, T, S)
                top_attn = attn_weights[-1].detach().cpu()  # tensor
                # 1 バッチ目のみ保存 (B==1 前提だが汎用に対応)
                att_map = top_attn[0].mean(0).numpy()       # (T, S)

                Path(save_dir).mkdir(parents=True, exist_ok=True)
                fig = plt.figure(figsize=(6, 4))
                plt.imshow(att_map, aspect="auto", origin="lower", cmap="viridis")
                plt.colorbar()
                plt.title(f"Epoch {epoch}  Batch {batch_idx}")
                plt.xlabel("Source (S)")
                plt.ylabel("Target (T)")
                out_path = (
                    Path(save_dir)
                    / f"ep{epoch:03d}_b{batch_idx:04d}.png"
                )
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                # 学習を止めないため、例外は握りつぶしてログだけ出す
                print(f"[Warn] attention save failed: {e}")

        return y, loss, mloss, diag_loss, ga_loss
   
    @torch.no_grad()
    def greedy_decode(
        self,
        src_mel: torch.Tensor,
        src_f0:  torch.Tensor,
        src_len: int,
        max_len: int = 200
    ) -> torch.Tensor:    
        # batch size must be 1
        assert src_mel.size(0) == 1
        device = src_mel.device
        S      = src_mel.size(1)
        D      = src_mel.size(-1)        

        # ── エンコーダ: use_f0=True のときのみ src_f0 を加算 ──
        prenet_src = self.prenet(src_mel)  # (1, S, D)
        if self.use_f0:
            p_src = self.pitch_proj(src_f0.unsqueeze(-1))  # (1, S, D)
            y     = self.pos_enc(prenet_src + p_src)
        else:
            p_src = None
            y     = self.pos_enc(prenet_src)        
        for layer in self.encoder_layers:
            y = layer(y, p_src)
        memory = y

        # 6. Initialize with trainable BOS vector
        ys = self.bos.expand(1, 1, D).clone()        
        f0_seq     = torch.zeros((1, 1),      device=device)  # (1,1) 最初は BOS のみ
        memory_mask = torch.zeros(self.nhead, 1, S, device=device)  # (H,1,S)
       

        for i in range(max_len - 1):
            # causal mask for current length
            T_cur = ys.size(1)
            mask  = self.generate_square_subsequent_mask(T_cur).to(device)

            # ── デコーダ入力: prenet 出力に必要なら f0 を加算 ──
            prenet_dec = self.prenet(ys)  # (1, T_cur, D)
            if self.use_f0:
                # これまでに予測した f0_seq をプロジェクト
                p_dec = self.pitch_proj(f0_seq.unsqueeze(-1))  # (1, T_cur, D)
                z     = self.pos_enc(prenet_dec + p_dec)
            else:
                p_dec = None
                z     = self.pos_enc(prenet_dec)            

            for layer in self.decoder_layers:
                z, _ = layer(
                    z,               # mel features
                    p_dec,           # f0 features (ground truth prefix or zero)                
                    memory,
                    tgt_mask=mask,
                    #memory_mask=None
                    memory_mask=memory_mask
                )        
            # postnet refinement
            z = self.postnet(z)
            # append last frame
            last = z[:, -1:, :]
            ys = torch.cat((ys, last), dim=1)        

            # ── 次ステップ用に f0 を予測してシーケンスに追加 ──
            if self.use_f0:
                # last: (1,1,D) → squeeze→(1,D) → out_f0→(1,1)
                f0_pred = self.out_f0(last.squeeze(1))
                f0_seq  = torch.cat((f0_seq, f0_pred.unsqueeze(1)), dim=1)
            
            # aggregate attention weights from all decoder layers
            weights = None
            for w in self.get_attention_weight():
                weights = w if weights is None else weights + w
            # weights: (B, num_heads, T_tgt, T_src)
            # select batch element 0 and average over heads
            att       = weights[0]               # (H, T_tgt, T_src)
            att_mean  = att.mean(dim=0)          # (T_tgt, T_src)
            last_att  = att_mean[-1]             # (T_src,)

            # ── 最も強く attend された整数位置 ──
            pos_int   = torch.argmax(last_att).item()

            # ── 単調性を保証：後退を禁止し、最大 +10 フレームまで前進 ──
            if i == 0:
                max_pos = pos_int
            else:
                prev    = max_pos
                max_pos = int(
                    torch.clamp(
                        torch.tensor(pos_int, device=device),
                        min=prev,          # 後ろに戻さない
                        max=prev + 10      # 急激に進みすぎない
                    ).item()
                )

            # デバッグ用の実数位置（参考値）
            pos_float = float(pos_int)            
                            
            # build new mask row for monotonic window [max_pos-8, max_pos+12)
            start = max(0, max_pos - 8)
            end   = min(src_len, max_pos + 12)

            # --- 新しい 1 行マスクを作成 ---
            mask_row = torch.full((self.nhead, 1, S), NEG_INF, device=device)  # すべて -inf
            mask_row[:, 0, start:end] = 0.0                                    # 窓内を 0.0 に

            # ── メモリマスクを (nhead , T_cur , S) へ拡張 ──
            # 既存マスクに 1 行追加して行数を T_cur と一致させる
            memory_mask = torch.cat((memory_mask, mask_row), dim=1)  # (H, T_cur , S)            
            
            # デバッグ出力（必要ならコメントアウト可）
            print(
                f"[step {i:03d}] "
                f"T_cur={ys.size(1):3d} "
                f"argmax={pos_float:.1f} → clipped={max_pos:3d} "
                f"window=[{start}:{end}]"
            )
            torch.cuda.empty_cache()

        return ys

    def _diagonal_attention_loss(self, attn_weights_list, ideal_weights, src_len, tgt_len):
        """
        各デコーダレイヤーから平均化した attention weights と
        理想の対角重みを用いて L1 ノルムで損失を計算する。
        attn_weights_list: List[Tensor] of shape (B, T_tgt, T_src)
        ideal_weights:     Tensor of shape (B, T_tgt, T_src)
        src_len, tgt_len:  LongTensor of shape (B,)
        """
        # ---- 各デコーダレイヤー → 合計して平均 (B,H,T,S) ----
        num_layers = len(attn_weights_list)
        w_sum = sum(attn_weights_list)               # (B, H, T, S)
        w_avg = w_sum / num_layers                   # (B, H, T, S)

        B, H, _, _ = w_avg.shape
        total_loss = 0.0
        for b in range(B):
            t = tgt_len[b].item()
            s = src_len[b].item()

            # ---- スライス: (H, t, s) ----
            w_slice = w_avg[b, :, :t, :s]            # (H, t, s)

            # 理想対角 (t,s) → (H,t,s) にブロードキャスト
            ideal_slice = ideal_weights[b, :t, :s].unsqueeze(0).expand_as(w_slice)

            total_loss += F.l1_loss(w_slice, ideal_slice)

        # バッチとヘッド数で正規化
        return total_loss / (B * H)

    def get_attention_weight(self):
        """
        各 DecoderLayerAlign が保持する直近の attention weights を取得して返す。
        Returns:
            List[Tensor]: 各要素は (B, num_heads, T_tgt, T_src) の attention weight
        """
        return [layer.last_attn for layer in self.decoder_layers]    
