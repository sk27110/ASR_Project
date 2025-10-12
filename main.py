from datasets import load_dataset

# Укажите корректные имена конфигурации и сплита
dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.100")
