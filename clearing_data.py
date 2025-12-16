# clean_text.py
import re
import pandas as pd

def minimal_clean(text):
    """
    Минимальная очистка для BERT:
    - Удаляем HTML-теги
    - Нормализуем пробелы
    - Оставляем всё остальное: пунктуацию, регистр, сокращения
    """
    if pd.isna(text):
        return ""
    # Удаляем HTML-теги
    text = re.sub(r'<[^>]+>', ' ', text)
    # Убираем множественные пробелы и обрезаем
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Применяем к датасету
DATA_PATH = r'N:\PyCharmRepo\NN_course\IMDB Dataset.csv'
OUTPUT_PATH = r'N:\PyCharmRepo\NN_course\IMDB_Dataset_BERT_ready.csv'

df = pd.read_csv(DATA_PATH)
df['bert_input'] = df['review'].apply(minimal_clean)  # используем исходный 'review'
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Данные сохранены в: {OUTPUT_PATH}")