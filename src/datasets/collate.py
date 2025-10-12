import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from dataset.__getitem__.

    Returns:
        result_batch (dict[str, torch.Tensor | list]): dict with batch data.
    """

    # === 1. Спектрограммы ===
    spectrograms = []
    for item in dataset_items:
        spec = item["spectrogram"]

        # если есть лишний канал → убираем
        if spec.dim() == 3 and spec.shape[0] == 1:
            spec = spec.squeeze(0)  # (n_mels, time)

        # убедимся, что получили (n_mels, time)
        assert spec.dim() == 2, f"Unexpected spectrogram shape: {spec.shape}"

        spectrograms.append(spec)

    spectrogram_lengths = torch.tensor(
        [spec.shape[1] for spec in spectrograms], dtype=torch.long
    )

    # Переставляем оси → (time, n_mels), чтобы pad_sequence работал
    spectrograms = [spec.transpose(0, 1) for spec in spectrograms]

    # Паддим по времени
    spectrograms_padded = pad_sequence(
        spectrograms, batch_first=True
    )  # (batch, max_time, n_mels)

    # Возвращаем обратно в (batch, n_mels, max_time)
    spectrograms_padded = spectrograms_padded.transpose(1, 2)

    # === 2. Тексты ===
    text_encoded = []
    for item in dataset_items:
        tensor = torch.tensor(item["text_encoded"], dtype=torch.long)

        # Если тензор имеет форму (1, N), превращаем в (N,)
        if tensor.dim() == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # убираем первую размерность
        elif tensor.dim() > 1:
            tensor = tensor.flatten()  # выравниваем полностью

        text_encoded.append(tensor)

    text_encoded_lengths = torch.tensor(
        [len(t) for t in text_encoded], dtype=torch.long
    )

    text_encoded_padded = pad_sequence(text_encoded, batch_first=True, padding_value=0)

    texts = [item["text"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]

    batch = {
        "spectrogram": spectrograms_padded,  # (batch, n_mels, max_time)
        "spectrogram_length": spectrogram_lengths,  # (batch,)
        "text_encoded": text_encoded_padded,  # (batch, max_text_len)
        "text_encoded_length": text_encoded_lengths,  # (batch,)
        "text": texts,  # list[str]
        "audio_path": audio_paths,  # list[str]
    }

    return batch
