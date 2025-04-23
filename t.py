import os
import shutil
import pandas as pd

# Путь к CSV-файлу с разметкой
csv_file = r"C:\Users\USER\Desktop\generated_synth\generated_synth\marking.csv"

# Папка, где локально находятся изображения.
# Например, если изображения лежат в local_images/images_handwritten/...
local_images_base = r"C:\Users\USER\Desktop\generated_synth\generated_synth\test"  # измените на ваш путь

# Читаем CSV
df = pd.read_csv(csv_file)

def process_subset(stage_value, dest_folder):
    """
    stage_value: значение в столбце 'stage' из CSV ("train" или "valid")
    dest_folder: имя папки для сохранения ("train" или "test")
    """
    # Папка для изображений внутри конечной структуры
    dest_img_dir = os.path.join("data", dest_folder, "test")
    gt_file_path = os.path.join("data", dest_folder, "gt.txt")
    
    # Создаём директории, если их ещё нет
    os.makedirs(dest_img_dir, exist_ok=True)
    
    # Фильтруем строки для нужного этапа
    subset_df = df[df["stage"] == stage_value]
    
    # Открываем файл gt.txt для записи
    with open(gt_file_path, "w", encoding="utf-8") as gt_file:
        for _, row in subset_df.iterrows():
            # Получаем исходный путь из CSV и удаляем префикс "reports/", если он есть
            src_path = row["path"]
            if src_path.startswith("reports/"):
                src_path = src_path[len("reports/"):]
            # Формируем полный путь к изображению, используя локальную базовую папку
            src_img_path = os.path.join(local_images_base, src_path)
            
            # Получаем имя файла (например, 10_9.jpg)
            filename = os.path.basename(src_img_path)
            label = row["text"]
            
            # Путь, куда копируется изображение
            dest_img_path = os.path.join(dest_img_dir, filename)
            try:
                shutil.copy2(src_img_path, dest_img_path)
            except FileNotFoundError:
                print(f"Файл не найден: {src_img_path}")
                continue
            
            # Записываем строку в gt.txt (путь относительно папки внутри dest_folder)
            relative_path = os.path.join("test", filename)
            gt_file.write(f"{relative_path}\t{label}\n")

# Обработка обучающей выборки (stage == "train" -> папка "train")
process_subset("train", "train")
# Обработка валидационной выборки (stage == "valid" -> папка "test")
process_subset("valid", "test")