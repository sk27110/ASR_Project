import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSpeechModel(nn.Module):
    """Быстрая но эффективная модель"""

    def __init__(self, n_feats, n_tokens, hidden=128, dropout=0.2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_feats, hidden, 3, padding=1),  # n_feats = 128
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, stride=2, padding=1),  # downsampling
            nn.ReLU(),
        )

        # Один GRU слой вместо LSTM
        self.gru = nn.GRU(hidden, hidden, 1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden * 2, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        # Вход: [batch, features, time] = [32, 128, 1617]
        # НЕ транспонируем - оставляем как есть для Conv1d
        # Conv1d ожидает: [batch, channels, length] = [32, 128, 1617]

        x = spectrogram  # уже в правильной форме [32, 128, 1617]
        x = self.conv(x)  # [32, hidden, new_length]

        # Транспонируем для GRU: [batch, channels, time] -> [batch, time, channels]
        x = x.transpose(1, 2)  # [32, new_length, hidden]

        x, _ = self.gru(x)
        log_probs = F.log_softmax(self.classifier(x), dim=-1)

        return {
            "log_probs": log_probs,
            "log_probs_length": self.transform_input_lengths(spectrogram_length),
        }

    def transform_input_lengths(self, input_lengths):
        """
        Calculate output lengths after conv layers with stride=2
        """
        return (input_lengths + 1) // 2

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        result_info = (
            result_info + f"\nInput shape: [batch, {self.conv[0].in_channels}, time]"
        )

        return result_info
