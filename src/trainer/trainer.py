from pathlib import Path

import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class for supervised training with AMP support.

    This class implements a training loop for a model, supporting:
    - Mixed precision training (autocast + GradScaler)
    - Metric computation for both train and inference
    - Logging of spectrograms and predictions to the writer
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Trainer.

        Args:
            *args, **kwargs: Arguments passed to BaseTrainer.
        """
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()  # AMP gradient scaler

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Process a single batch: forward pass, backward pass, update metrics.

        Args:
            batch (dict): Batch data from the DataLoader.
            metrics (MetricTracker): Metric tracker for updating losses/metrics.

        Returns:
            dict: Batch updated with model outputs and loss values.
        """
        # Move batch to device and apply batch transforms
        batch = self.move_batch_to_device(batch, non_blocking=True)
        batch = self.transform_batch(batch)

        # Select metrics based on train/inference mode
        metric_funcs = (
            self.metrics["inference"] if not self.is_train else self.metrics["train"]
        )

        if self.is_train:
            self.optimizer.zero_grad()

        # --- Forward pass with mixed precision ---
        with autocast():
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        # --- Backward pass with GradScaler ---
        if self.is_train:
            self.scaler.scale(batch["loss"]).backward()
            self._clip_grad_norm()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # --- Update metrics ---
        for loss_name in getattr(self.config.writer, "loss_names", []):
            metrics.update(loss_name, batch[loss_name].item())
        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log batch information: spectrograms and predictions.

        Args:
            batch_idx (int): Index of the current batch.
            batch (dict): Batch data after processing.
            mode (str): "train" or "inference" mode.
        """
        self.log_spectrogram(**batch)
        if mode != "train":
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        """
        Log the first spectrogram in the batch.

        Args:
            spectrogram (Tensor): Batch of spectrograms.
        """
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        """
        Log model predictions, calculate WER/CER, and create a table.

        Args:
            text (list[str] | Tensor): Ground-truth transcripts.
            log_probs (Tensor): Model log probabilities output.
            log_probs_length (Tensor): Lengths of log_probs sequences.
            audio_path (list[str]): Paths of audio files.
            examples_to_log (int): Maximum number of examples to log.
        """
        # Decode text if provided as tensor
        if torch.is_tensor(text):
            text = [self.text_encoder.decode(t) for t in text]

        # Select decoding mode
        decode_mode = getattr(self.config.decode, "mode", "argmax")
        if decode_mode == "beam_search":
            pred_texts = self.text_encoder.beam_search_decode(
                log_probs, log_probs_length
            )
        elif decode_mode == "prefix_beam_search":
            pred_texts = self.text_encoder.ctc_prefix_beam_search_decode(
                log_probs, log_probs_length
            )
        else:
            argmax_inds = torch.argmax(log_probs, dim=-1)
            pred_texts = [
                self.text_encoder.ctc_decode(inds[:l])
                for inds, l in zip(argmax_inds, log_probs_length)
            ]

        # Raw argmax decoding
        argmax_inds = torch.argmax(log_probs, dim=-1)
        raw_texts = [
            self.text_encoder.ctc_decode(inds[:l])
            for inds, l in zip(argmax_inds, log_probs_length)
        ]

        # Prepare table for logging
        tuples = list(zip(pred_texts, text, raw_texts, audio_path))
        rows = {}
        for pred, target, raw_pred, path in tuples[:examples_to_log]:
            target_norm = self.text_encoder.normalize_text(target)
            wer = calc_wer(target_norm, pred) * 100
            cer = calc_cer(target_norm, pred) * 100

            rows[Path(path).name] = {
                "target": target_norm,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
            }

        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
