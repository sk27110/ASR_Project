import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechModel(nn.Module):
    """
    DeepSpeech2-подобная модель с улучшениями для LibriSpeech 100h.
    - Conv frontend: 2 слоя, 64→128 каналов
    - RNN backend: bidirectional GRU + LayerNorm
    - Dropout для регуляризации
    - Выход: log_probs для CTC
    """

    def __init__(
        self, n_feats=128, n_tokens=29, rnn_hidden=512, num_rnn_layers=5, dropout=0.1
    ):
        super().__init__()

        # --- Conv frontend ---
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Hardtanh(0, 20, inplace=True),
        )

        # Conv output features
        self._n_feats_out = n_feats // (2 * 2)  # stride=2 дважды

        # --- RNN backend ---
        rnn_input_size = 128 * self._n_feats_out
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.rnn_layernorm = nn.LayerNorm(rnn_hidden * 2)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, n_tokens),
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Args:
            spectrogram: [B, n_feats, T]
        Returns:
            dict with log_probs [B, T', n_tokens] and log_probs_length
        """
        # Добавляем channel dimension
        x = spectrogram.unsqueeze(1)  # [B, 1, n_feats, T]

        # --- Conv frontend ---
        x = self.conv(x)  # [B, 128, n_feats', T']

        # Подготовка к RNN
        B, C, L, T = x.size()
        x = x.permute(0, 3, 1, 2)  # [B, T, C, L]
        x = x.contiguous().view(B, T, C * L)  # [B, T, features]

        # --- RNN ---
        x, _ = self.rnn(x)
        x = self.rnn_layernorm(x)

        # --- Classifier ---
        logits = self.classifier(x)
        log_probs = F.log_softmax(logits, dim=-1)

        # Длина последовательности после Conv
        out_lengths = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": out_lengths}

    def transform_input_lengths(self, input_lengths):
        # два Conv слоя со stride=2 → длина уменьшается в 4 раза
        for _ in range(2):
            input_lengths = (input_lengths + 1) // 2
        return input_lengths

    def __str__(self):
        all_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info = super().__str__()
        info += f"\nAll parameters: {all_params: n}"
        info += f"\nTrainable parameters: {trainable: n}"
        info += f"\nConv output feats: {self._n_feats_out}"

        return info
