import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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

# --- Diagonal Bias Mask ---
def compute_diagonal_mask(T: int, S: int, nu: float = 0.3, device=None) -> torch.Tensor:
    tgt = torch.arange(T, device=device).unsqueeze(1).float() / (T - 1)
    src = torch.arange(S, device=device).unsqueeze(0).float() / (S - 1)
    diff = (src - tgt) ** 2
    weight = torch.exp(-diff / (2 * nu * nu))
    mask = torch.log(weight)
    mask = torch.clamp(mask, min=-1e9)

    bad_row = (mask == -1e9).all(dim=-1)
    if bad_row.any():
        rows = bad_row.nonzero(as_tuple=False).squeeze(1)
        cols = ((src - tgt).abs().argmin(dim=-1))[rows]
        mask[rows, cols] = 0.0

    return mask

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
        diag_weight: float = 1.0,
        ce_weight: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.nu = nu
        self.diag_weight = diag_weight
        self.ce_weight = ce_weight

        # Projections
        self.mel_proj   = nn.Linear(input_dim_mel, d_model)
        self.pitch_proj = nn.Linear(input_dim_pitch, d_model)
        self.posenc     = FixedPositionalEncoding(d_model, max_len=8000)

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder_layers.append(nn.ModuleDict({
                'self_attn':  nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'pitch_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'ffn':        nn.Sequential(
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
                'self_attn':  nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'pitch_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'cross_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'ffn':        nn.Sequential(
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
        self.bos_token        = nn.Parameter(torch.randn(1,1,d_model))
        self.eos_token        = nn.Parameter(torch.randn(1,1,d_model))

    def forward(self,
                src_mel, src_pitch, src_lengths,
                tgt_mel, tgt_pitch, tgt_lengths):
        B, S, _ = src_mel.size()
        _, T, _ = tgt_mel.size()
        device  = src_mel.device
        
        # パディングマスク
        # src_pad_mask[b, i] = True if frame i is padding
        src_pad_mask = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        src_pad_mask = src_pad_mask >= src_lengths.unsqueeze(1)
        # tgt 用は bos を含むので長さは T+1
        tgt_pad_mask = torch.arange(T+1, device=device).unsqueeze(0).expand(B, T+1)
        tgt_pad_mask = tgt_pad_mask >= (tgt_lengths + 1).unsqueeze(1)

        # ─── Encoder ──
        x = self.mel_proj(src_mel) + self.pitch_proj(src_pitch.unsqueeze(-1))
        x = self.posenc(x)
        p_enc = self.pitch_proj(src_pitch.unsqueeze(-1))

        for layer in self.encoder_layers:
            # self-attn with src_pad_mask
            x2, _ = layer['self_attn'](
                x, x, x,
                key_padding_mask=src_pad_mask,
                need_weights=False)
            x = layer['ffn'](x + x2)

            # pitch-attn with same mask
            x2p, _ = layer['pitch_attn'](
                x, p_enc, p_enc,
                key_padding_mask=src_pad_mask,
                need_weights=False)
            x = layer['ffn'](x + x2p)
        memory = x

        # ─── Decoder i
        bos = self.bos_token.expand(B, 1, self.d_model)
        t_h = self.mel_proj(tgt_mel)
        t_p = self.pitch_proj(tgt_pitch.unsqueeze(-1))
        decoder_input = torch.cat([bos, t_h + t_p], dim=1)
        decoder_input = self.posenc(decoder_input)

        p_dec = torch.cat([torch.zeros(B,1,self.d_model, device=device), t_p], dim=1)
        # 対角マスクは (T+1, S)
        mask = compute_diagonal_mask(T+1, S, self.nu, device)

        # ─── Decoder ─
        x_dec = decoder_input
        for layer in self.decoder_layers:
            # causal self-attn (tgt→tgt) に tgt_pad_mask
            causal_mask = torch.triu(
                torch.full((x_dec.size(1), x_dec.size(1)), float('-inf'), device=device),
                diagonal=1)
            y2, _ = layer['self_attn'](
                x_dec, x_dec, x_dec,
                attn_mask=causal_mask,
                key_padding_mask=tgt_pad_mask,
                need_weights=False)
            y = layer['ffn'](x_dec + y2)

            # pitch-attn
            y2p, _ = layer['pitch_attn'](
                y, p_dec, p_dec,
                need_weights=False)
            y = layer['ffn'](y + y2p)

            # cross-attn with both masks
            bool_mask = (mask == float('-inf'))
            y2m, attn_w = layer['cross_attn'](
                y, memory, memory,
                attn_mask=bool_mask,
                key_padding_mask=src_pad_mask,
                need_weights=True)
            x_dec = layer['ffn'](y + y2m)

        # ─── 出力・損失
        pred_mel   = self.out_mel(x_dec)
        pred_pitch = self.out_pitch(x_dec).squeeze(-1)
        logits     = self.token_classifier(x_dec)

        # パディングを除いた損失計算
        loss_mel = 0
        loss_p   = 0
        for b in range(B):
            L = tgt_lengths[b].item() + 1  # bos 含む
            loss_mel += F.l1_loss(pred_mel[b, :L], torch.cat([tgt_mel[b], torch.zeros(1, tgt_mel.size(-1), device=device)]))
            loss_p   += F.l1_loss(pred_pitch[b, :L], torch.cat([tgt_pitch[b].unsqueeze(0), torch.zeros(1, device=device)]))
        loss_mel /= B
        loss_p   /= B

        # 以下、loss_diag, loss_ce は変更なし
        total = loss_mel + loss_p + self.diag_weight*loss_diag + self.ce_weight*loss_ce
        return total, {'mel_l1': loss_mel, 'pitch_l1': loss_p, 'diag': loss_diag, 'ce': loss_ce}

    '''
    def forward(self, src_mel, src_pitch, tgt_mel, tgt_pitch):
        B, S, _ = src_mel.size()
        _, T, _ = tgt_mel.size()
        device  = src_mel.device

        # Encoder
        x = self.mel_proj(src_mel) + self.pitch_proj(src_pitch.unsqueeze(-1))
        x = self.posenc(x)
        p_enc = self.pitch_proj(src_pitch.unsqueeze(-1))

        def enc_block(x, p_stream, layer):
            x2, _ = layer['self_attn'](x, x, x, need_weights=False)
            x_    = layer['ffn'](x + x2)
            x2p, _ = layer['pitch_attn'](x_, p_stream, p_stream, need_weights=False)
            return layer['ffn'](x_ + x2p)

        for layer in self.encoder_layers:
            #x = checkpoint(enc_block, x, p_enc, layer)
            x = enc_block(x, p_enc, layer)
        memory = x

        # Decoder init
        bos = self.bos_token.expand(B, 1, self.d_model)
        t_h = self.mel_proj(tgt_mel)
        t_p = self.pitch_proj(tgt_pitch.unsqueeze(-1))
        decoder_input = torch.cat([bos, t_h + t_p], dim=1)
        decoder_input = self.posenc(decoder_input)

        p_dec = torch.cat([torch.zeros(B,1,self.d_model, device=device), t_p], dim=1)
        mask = compute_diagonal_mask(decoder_input.size(1), S, self.nu, device)

        def dec_block(x, p_stream, memory, layer, mask):
            y2, _ = layer['self_attn'](x, x, x, need_weights=False)
            y_     = layer['ffn'](x + y2)
            y2p, _ = layer['pitch_attn'](y_, p_stream, p_stream, need_weights=False)
            y__    = layer['ffn'](y_ + y2p)
            y2m, attn = layer['cross_attn'](y__, memory, memory, attn_mask=mask, need_weights=True)
            return layer['ffn'](y__ + y2m), attn

        attn_w = None
        x_dec = decoder_input
        for layer in self.decoder_layers:
            x_dec, attn_w = dec_block(x_dec, p_dec, memory, layer, mask)
            #x_dec, attn_w = checkpoint(dec_block, x_dec, p_dec, memory, layer, mask)

        # Outputs
        pred_mel   = self.out_mel(x_dec)
        pred_pitch = self.out_pitch(x_dec).squeeze(-1)
        logits     = self.token_classifier(x_dec)

        # Loss
        tgt_mel_pad = torch.cat([tgt_mel, torch.zeros(B,1,tgt_mel.size(-1),device=device)], dim=1)
        tgt_p_pad   = torch.cat([tgt_pitch.unsqueeze(-1), torch.zeros(B,1,1,device=device)], dim=1).squeeze(-1)

        loss_mel = F.l1_loss(pred_mel, tgt_mel_pad)
        loss_p   = F.l1_loss(pred_pitch, tgt_p_pad)

        # Diagonal loss
        pos_s = torch.arange(S, device=device).unsqueeze(0).repeat(T+1,1)
        pos_t = torch.arange(T+1, device=device).unsqueeze(1).repeat(1,S)
        dist  = torch.abs(pos_t - pos_s).float()/S
        loss_diag = (attn_w * dist.unsqueeze(0)).sum()/(B*(T+1))

        # EOS CE loss
        labels = torch.zeros(B, T+1, dtype=torch.long, device=device)
        labels[:,-1] = 1
        logits_clamped = torch.clamp(logits, -20.0, 20.0)
        loss_ce = F.cross_entropy(logits_clamped.view(-1,2), labels.view(-1), label_smoothing=0.1)

        total = loss_mel + loss_p + self.diag_weight*loss_diag + self.ce_weight*loss_ce
        return total, {'mel_l1': loss_mel, 'pitch_l1': loss_p, 'diag': loss_diag, 'ce': loss_ce}
    '''
    
    def greedy_decode(self, src_mel, src_pitch, max_len=200):
        """
        Greedy decode using log-mel spectrogram and pitch with causal mask and dynamic monotonic control.
        Returns mel_seq: (B, T, F), pitch_seq: (B, T)
        """
        B, S, _ = src_mel.size()    # S = メモリ長
        device = src_mel.device

        # --- Encode ---
        x = self.mel_proj(src_mel) + self.pitch_proj(src_pitch.unsqueeze(-1))
        x = self.posenc(x)
        for layer in self.encoder_layers:
            x2, _ = layer['self_attn'](x, x, x)
            x = layer['ffn'](x + x2)
            p = self.pitch_proj(src_pitch.unsqueeze(-1))
            x2p, _ = layer['pitch_attn'](x, p, p)
            x = layer['ffn'](x + x2p)
        memory = x

        # --- Decode loop ---
        current = self.bos_token.expand(B, 1, self.d_model)
        decoded_m, decoded_p = [], []
        decoded_p_embed = []       # pitch 埋め込みを時系列でためておく
        memory_mask = None
        ended = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_len):
            y = current

            # 1) pitch_stream を組み立て
            if decoded_p_embed:
                p_stream = torch.cat(decoded_p_embed, dim=1)  # (B, t, D)
            else:
                p_stream = torch.zeros(B, 1, self.d_model, device=device)

            # 各デコーダ層を通過
            for layer in self.decoder_layers:
                # causal self-attention
                T_y = y.size(1)
                causal_mask = torch.triu(torch.full((T_y, T_y), float('-inf'), device=device), diagonal=1)
                y2, _ = layer['self_attn'](y, y, y, attn_mask=causal_mask)
                y = layer['ffn'](y + y2)

                # pitch cross-attention
                y2p, _ = layer['pitch_attn'](y, p_stream, p_stream)
                y = layer['ffn'](y + y2p)

                # diagonal + dynamic monotonic cross-attention
                diag_mask = compute_diagonal_mask(T_y, S, self.nu, device)  # (T_y, S)

                if memory_mask is not None:
                    prev_T, _ = memory_mask.shape  # 前ステップの (T_prev, S)
                    cur_T = T_y
                    # 前の mask を今の長さに合わせてパディング or トリム
                    if prev_T < cur_T:
                        pad = torch.full((cur_T - prev_T, S), float('-inf'), device=device)
                        mem = torch.cat([memory_mask, pad], dim=0)
                    elif prev_T > cur_T:
                        mem = memory_mask[-cur_T:, :]
                    else:
                        mem = memory_mask
                    combined = diag_mask + mem
                else:
                    combined = diag_mask

                bool_mask = (combined == float('-inf'))
                y2m, attn_w = layer['cross_attn'](y, memory, memory, attn_mask=bool_mask, need_weights=True)
                y = layer['ffn'](y + y2m)

                # 動的マスクを更新（最後の frame の attention 位置に基づくウィンドウ）
                last_attn = attn_w[:, -1, :].mean(dim=0)  # (S,)
                pos = torch.argmax(last_attn).item()
                start = max(pos - 5, 0)
                end   = min(pos + 10, S)
                m_mask = torch.full((T_y, S), float('-inf'), device=device)
                m_mask[:, start:end] = 0.0
                memory_mask = m_mask

            # --- フレーム予測 & 次ステップ準備 ---
            last    = y[:, -1, :]                    # (B, D)
            m_pred  = self.out_mel(last)             # (B, F)
            p_pred  = self.out_pitch(last).squeeze(-1)  # (B,)

            decoded_m.append(m_pred.unsqueeze(1))    # list of (B,1,F)
            decoded_p.append(p_pred.unsqueeze(1))    # list of (B,1)

            # pitch 埋め込みをためておく
            p1 = self.pitch_proj(p_pred.unsqueeze(-1))  # (B, D)
            decoded_p_embed.append(p1.unsqueeze(1))     # (B, t+1, D)

            # 次の input
            fused = self.mel_proj(m_pred) + p1
            fused = self.posenc(fused.unsqueeze(1), start=current.size(1)).squeeze(1)
            current = torch.cat([current, fused.unsqueeze(1)], dim=1)

            # EOS チェック
            ended |= (self.token_classifier(last).argmax(-1) == 1)
            if ended.all():
                break

        mel_seq   = torch.cat(decoded_m, dim=1)  # (B, T, F)
        pitch_seq = torch.cat(decoded_p, dim=1)  # (B, T)
        return mel_seq, pitch_seq


    '''
    def greedy_decode(self, src_mel, src_pitch, max_len=200):
        """
        Greedy decode using log-mel spectrogram and pitch with causal mask and dynamic monotonic control.
        Returns mel_seq: (B, T, F), pitch_seq: (B, T)
        """
        B, S, _ = src_mel.size()
        device = src_mel.device

        # --- Encode ---
        x = self.mel_proj(src_mel) + self.pitch_proj(src_pitch.unsqueeze(-1))
        x = self.posenc(x)
        for layer in self.encoder_layers:
            x2, _ = layer['self_attn'](x, x, x)
            x = layer['ffn'](x + x2)
            p = self.pitch_proj(src_pitch.unsqueeze(-1))
            x2p, _ = layer['pitch_attn'](x, p, p)
            x = layer['ffn'](x + x2p)
        memory = x

        # prepare monotonic mask placeholder
        memory_mask = None

        # --- Decode loop ---
        current = self.bos_token.expand(B, 1, self.d_model)
        decoded_m, decoded_p = [], []
        ended = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_len):
            y = current
            for layer in self.decoder_layers:
                # 1) causal self-attention
                T_y = y.size(1)
                causal_mask = torch.triu(torch.full((T_y, T_y), float('-inf'), device=device), diagonal=1)
                y2, _ = layer['self_attn'](y, y, y, attn_mask=causal_mask)
                y = layer['ffn'](y + y2)

                # 2) pitch cross-attention
                p_stream = torch.cat([
                    torch.zeros(B, 1, self.d_model, device=device),
                    torch.cat(decoded_p, dim=1).unsqueeze(-1) if decoded_p else torch.zeros(B, 1, 1, device=device)
                ], dim=1)
                y2p, _ = layer['pitch_attn'](y, p_stream, p_stream)
                y = layer['ffn'](y + y2p)

                # 3) diagonal + dynamic monotonic cross-attention
                diag_mask = compute_diagonal_mask(y.size(1), S, self.nu, device)
                combined = diag_mask
                if memory_mask is not None:
                    combined = combined + memory_mask
                bool_mask = (combined == float('-inf'))
                y2m, attn_w = layer['cross_attn'](y, memory, memory, attn_mask=bool_mask, need_weights=True)
                y = layer['ffn'](y + y2m)

                # update dynamic mask based on last attention
                last_attn = attn_w[:, -1, :].mean(dim=0)  # (S,)
                pos = torch.argmax(last_attn).item()
                start = max(pos - 5, 0)
                end = min(pos + 10, S)
                # build new mask for next step
                m_mask = torch.full((y.size(1), S), float('-inf'), device=device)
                m_mask[:, start:end] = 0.0
                memory_mask = m_mask

            # predict last frame
            last = y[:, -1, :]
            m_pred = self.out_mel(last)           # (B, F)
            p_pred = self.out_pitch(last).squeeze(-1)  # (B,)
            decoded_m.append(m_pred.unsqueeze(1))
            decoded_p.append(p_pred.unsqueeze(1))

            # prepare next input
            fused = self.mel_proj(m_pred) + self.pitch_proj(p_pred.unsqueeze(-1))
            fused = self.posenc(fused.unsqueeze(1), start=current.size(1)).squeeze(1)
            current = torch.cat([current, fused.unsqueeze(1)], dim=1)

            # check EOS
            ended |= (self.token_classifier(last).argmax(-1) == 1)
            if ended.all():
                break

        mel_seq = torch.cat(decoded_m, dim=1)
        pitch_seq = torch.cat(decoded_p, dim=1)
        return mel_seq, pitch_seq
    '''
    
    def predict(self, src_mel, src_pitch, max_len=200):
        """
        Wrapper around greedy_decode for inference.
        """
        return self.greedy_decode(src_mel, src_pitch, max_len)
