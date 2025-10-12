import os

import deeplake

# 1. Указываем директорию для кэша Deeplake
DATA_DIR = "./data/datasets/librispeech"

os.makedirs(DATA_DIR, exist_ok=True)
os.environ["DEEPLAKE_PATH"] = DATA_DIR

print("Загрузка датасета...")
ds = deeplake.load("hub://activeloop/LibriSpeech-train-clean-100")
print("✅ Датасет подключён")

# Принудительная загрузка всех элементов (триггер кэша)
print("📥 Скачиваем данные локально...")
for sample in ds:
    # Обращение к данным (например, аудио и транскрипт)
    _ = sample.audio.numpy()
    _ = sample.transcript.numpy()

print(f"✅ Все данные загружены и закэшированы в {DATA_DIR}")
