from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "prajjwal1/bert-small"
SAVE_PATH = r"N:\PyCharmRepo\NN_course\prajjwal1_bert_small"

print("Скачивание предобученной модели bert-small...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

print("Сохранение локально...")
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"✅ Готово! Модель сохранена в: {SAVE_PATH}")