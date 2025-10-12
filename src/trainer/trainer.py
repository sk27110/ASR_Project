from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    Supports argmax and beam search decoding for predictions.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        Args:
            batch (dict): batch from dataloader
            metrics (MetricTracker): metric tracker instance
        Returns:
            batch (dict): updated batch with outputs and losses
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = (
            self.metrics["inference"] if not self.is_train else self.metrics["train"]
        )

        if self.is_train:
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss
        for loss_name in getattr(self.config.writer, "loss_names", []):
            metrics.update(loss_name, batch[loss_name].item())

        # update task metrics
        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log spectrograms and predictions.
        """
        self.log_spectrogram(**batch)
        if mode != "train":
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        """
        Log a single spectrogram image.
        """
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image(
            "spectrogram", image
        )  # закомментировано, чтобы не было ошибок

    # В методе log_predictions в trainer.py
    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        # Convert text to list of strings if it's a tensor
        if torch.is_tensor(text):
            text = [self.text_encoder.decode(t) for t in text]

        decode_mode = getattr(self.config.decode, "mode", "argmax")

        if decode_mode == "beam_search":
            pred_texts = self.text_encoder.beam_search_decode(
                log_probs, log_probs_length
            )
        elif decode_mode == "prefix_beam_search":
            pred_texts = self.text_encoder.ctc_prefix_beam_search_decode(
                log_probs, log_probs_length
            )
        else:  # argmax
            argmax_inds = torch.argmax(log_probs, dim=-1)
            pred_texts = [
                self.text_encoder.ctc_decode(inds[:l])
                for inds, l in zip(argmax_inds, log_probs_length)
            ]

        # For comparison, also get raw argmax decoding
        argmax_inds = torch.argmax(log_probs, dim=-1)
        raw_texts = [
            self.text_encoder.ctc_decode(inds[:l])
            for inds, l in zip(argmax_inds, log_probs_length)
        ]

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
