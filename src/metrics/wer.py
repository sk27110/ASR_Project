# src/metrics/wer.py
from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer


class WERMetric(BaseMetric):
    def __init__(self, text_encoder, mode="argmax", beam_width=5, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.mode = mode
        self.beam_width = beam_width

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []

        if self.mode == "beam_search":
            preds = self.text_encoder.beam_search_decode(log_probs, log_probs_length)
            for pred_text, target_text in zip(preds, text):
                target_text = self.text_encoder.normalize_text(target_text)
                wers.append(calc_wer(target_text, pred_text))
        else:  # argmax
            predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
            lengths = log_probs_length.cpu().numpy()
            for log_prob_vec, length, target_text in zip(predictions, lengths, text):
                target_text = self.text_encoder.normalize_text(target_text)
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
                wers.append(calc_wer(target_text, pred_text))

        return sum(wers) / len(wers)
