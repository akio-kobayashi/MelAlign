import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class GreedyEvalCallbackMel(pl.Callback):
    """
    Performs greedy decoding on validation set every N epochs for Mel-to-Mel alignment.
    Metrics: L1 on mel and pitch, and total.
    """
    def __init__(self, every_n_epochs: int = 5, max_len: int = 200, num_batches: int = 1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.max_len = max_len
        self.num_batches = num_batches

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        # Get first validation DataLoader
        val_loaders = trainer.val_dataloaders
        val_loader = val_loaders[0] if isinstance(val_loaders, (list, tuple)) else val_loaders

        # Optionally average over multiple batches
        mel_losses = []
        pitch_losses = []
        for i, batch in enumerate(val_loader):
            if i >= self.num_batches:
                break
            src_m, src_p, tgt_m, tgt_p = [t.to(pl_module.device) for t in batch]
            m_pred, p_pred = pl_module.model.greedy_decode(src_m, src_p, max_len=self.max_len)

            # Pad/truncate mel and pitch
            T = tgt_m.size(1)
            if m_pred.size(1) < T:
                pad = T - m_pred.size(1)
                m_pred = torch.cat([m_pred, torch.zeros(m_pred.size(0), pad, m_pred.size(2), device=pl_module.device)], dim=1)
                p_pred = torch.cat([p_pred, torch.zeros(p_pred.size(0), pad, device=pl_module.device)], dim=1)
            else:
                m_pred = m_pred[:, :T]
                p_pred = p_pred[:, :T]

            mel_losses.append(F.l1_loss(m_pred, tgt_m).item())
            pitch_losses.append(F.l1_loss(p_pred, tgt_p).item())

        # Compute average losses
        avg_mel = sum(mel_losses) / len(mel_losses)
        avg_pitch = sum(pitch_losses) / len(pitch_losses)
        avg_total = avg_mel + avg_pitch

        metrics = {
            "greedy/l1_mel": avg_mel,
            "greedy/l1_pitch": avg_pitch,
            "greedy/l1_total": avg_total
        }
        trainer.logger.log_metrics(metrics, step=epoch)
