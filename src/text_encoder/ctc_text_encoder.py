# src/text/ctc_encoder.py
import re
from collections import defaultdict
from string import ascii_lowercase
from typing import List, Tuple, Union

import numpy as np
import torch


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, use_bpe=False, beam_width=5, **kwargs):
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.use_bpe = use_bpe
        self.beam_width = beam_width

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        """Encode text to tensor of indices"""
        if isinstance(text, list):
            # Handle batch encoding
            encoded = []
            for t in text:
                t_norm = self.normalize_text(t)
                try:
                    encoded.append([self.char2ind[c] for c in t_norm])
                except KeyError:
                    unknown_chars = set(
                        [char for char in t_norm if char not in self.char2ind]
                    )
                    raise Exception(
                        f"Can't encode text '{t}'. Unknown chars: '{' '.join(unknown_chars)}'"
                    )

            # Pad sequences to same length
            max_len = max(len(e) for e in encoded)
            padded = [
                e + [self.char2ind[self.EMPTY_TOK]] * (max_len - len(e))
                for e in encoded
            ]
            return torch.tensor(padded)
        else:
            # Single text encoding
            text = self.normalize_text(text)
            try:
                return torch.tensor([self.char2ind[c] for c in text]).unsqueeze(0)
            except KeyError:
                unknown_chars = set(
                    [char for char in text if char not in self.char2ind]
                )
                raise Exception(
                    f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
                )

    def decode(self, inds) -> str:
        """Direct decoding without CTC collapse"""
        if torch.is_tensor(inds):
            inds = inds.cpu().numpy()
        return "".join(
            [self.ind2char[int(i)] for i in inds if int(i) < len(self.ind2char)]
        ).strip()

    def ctc_decode(self, inds) -> str:
        """CTC decoding that collapses repeated characters and removes blanks"""
        if torch.is_tensor(inds):
            inds = inds.cpu().numpy()

        decoded = []
        last_char_ind = self.char2ind[self.EMPTY_TOK]

        for ind in inds:
            ind = int(ind)
            if ind == last_char_ind:
                continue
            if ind != self.char2ind[self.EMPTY_TOK]:
                decoded.append(self.ind2char[ind])
            last_char_ind = ind

        return "".join(decoded)

    def beam_search_decode(
        self, log_probs: torch.Tensor, log_probs_length: torch.Tensor
    ) -> List[str]:
        """
        Proper CTC beam search implementation.

        Args:
            log_probs: [B, T, V] log probabilities from model
            log_probs_length: [B] actual lengths of sequences

        Returns:
            List of decoded strings for each batch element
        """
        log_probs = log_probs.cpu().numpy()

        if torch.is_tensor(log_probs_length):
            lengths = log_probs_length.cpu().numpy()
        else:
            lengths = log_probs_length

        batch_results = []

        for batch_idx, length in enumerate(lengths):
            # Get actual sequence (remove padding)
            sequence = log_probs[batch_idx, :length, :]
            beams = [BeamPath(self.beam_width)]  # Start with empty path

            for timestep in sequence:
                new_beams = []

                for beam in beams:
                    # Expand each beam with all possible characters
                    for char_idx, log_prob in enumerate(timestep):
                        new_beam = beam.copy()
                        new_beam.add(char_idx, log_prob)
                        new_beams.append(new_beam)

                # Keep only top-k beams
                new_beams.sort(key=lambda x: x.score, reverse=True)
                beams = new_beams[: self.beam_width]

            # Get best beam and convert to text
            if beams:
                best_beam = beams[0]
                best_sequence = best_beam.get_sequence()
                pred_text = self.ctc_decode(torch.tensor(best_sequence))
            else:
                pred_text = ""

            batch_results.append(pred_text)

        return batch_results

    def ctc_prefix_beam_search_decode(
        self, log_probs: torch.Tensor, log_probs_length: torch.Tensor
    ) -> List[str]:
        """
        Alternative CTC prefix beam search implementation (more accurate).
        Based on: https://distill.pub/2017/ctc/
        """
        log_probs = log_probs.cpu().numpy()

        if torch.is_tensor(log_probs_length):
            lengths = log_probs_length.cpu().numpy()
        else:
            lengths = log_probs_length

        batch_results = []

        for batch_idx, length in enumerate(lengths):
            sequence = log_probs[batch_idx, :length, :]

            # Initialize with empty sequence
            beams = [("", 0.0)]  # (text, score)
            blank_idx = self.char2ind[self.EMPTY_TOK]

            for timestep in sequence:
                new_beams = {}

                for text, score in beams:
                    for char_idx, log_p in enumerate(timestep):
                        new_score = score + log_p

                        if char_idx == blank_idx:
                            # Blank - keep current text
                            new_beams[text] = max(
                                new_beams.get(text, -float("inf")), new_score
                            )
                        else:
                            char = self.ind2char[char_idx]

                            if text and text[-1] == char:
                                # Repeated character
                                new_text = text
                            else:
                                # New character
                                new_text = text + char

                            new_beams[new_text] = max(
                                new_beams.get(new_text, -float("inf")), new_score
                            )

                # Keep top-k beams
                beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[
                    : self.beam_width
                ]

            batch_results.append(beams[0][0] if beams else "")

        return batch_results

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text: lowercase and remove special characters"""
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def set_beam_width(self, beam_width: int):
        """Update beam width for decoding"""
        self.beam_width = beam_width


class BeamPath:
    """Helper class for beam search tracking"""

    def __init__(self, beam_width):
        self.sequence = []
        self.score = 0.0
        self.beam_width = beam_width

    def add(self, char_idx: int, log_prob: float):
        """Add character to beam path"""
        self.sequence.append(char_idx)
        self.score += log_prob

    def copy(self):
        """Create a copy of the beam path"""
        new_beam = BeamPath(self.beam_width)
        new_beam.sequence = self.sequence.copy()
        new_beam.score = self.score
        return new_beam

    def get_sequence(self):
        """Get the current sequence"""
        return self.sequence.copy()
