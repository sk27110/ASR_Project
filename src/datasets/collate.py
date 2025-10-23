import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Формирует батч из элементов датасета, выравнивая длины спектрограмм и текстов.
    """

    spectrograms = []
    for item in dataset_items:
        spec = item["spectrogram"]
        if spec.dim() == 3 and spec.shape[0] == 1:
            spec = spec.squeeze(0)
        assert spec.dim() == 2, f"Unexpected spectrogram shape: {spec.shape}"
        spectrograms.append(spec)

    spectrogram_lengths = torch.tensor(
        [spec.shape[1] for spec in spectrograms], dtype=torch.long
    )

    spectrograms = [spec.transpose(0, 1) for spec in spectrograms]
    spectrograms_padded = pad_sequence(spectrograms, batch_first=True)
    spectrograms_padded = spectrograms_padded.transpose(1, 2)

    text_encoded = []
    for item in dataset_items:
        tensor = torch.tensor(item["text_encoded"], dtype=torch.long)
        if tensor.dim() == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        elif tensor.dim() > 1:
            tensor = tensor.flatten()
        text_encoded.append(tensor)

    text_encoded_lengths = torch.tensor(
        [len(t) for t in text_encoded], dtype=torch.long
    )

    text_encoded_padded = pad_sequence(text_encoded, batch_first=True, padding_value=0)

    texts = [item["text"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]

    batch = {
        "spectrogram": spectrograms_padded,
        "spectrogram_length": spectrogram_lengths,
        "text_encoded": text_encoded_padded,
        "text_encoded_length": text_encoded_lengths,
        "text": texts,
        "audio_path": audio_paths,
    }

    return batch
