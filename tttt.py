import os
import json
import shutil
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image

# === ПАРАМЕТРЫ ===
CSV_PATH = r"C:\shared\reports15\annotations_with_image_size.csv"
IMG_DIR = r"C:\shared\reports15\combined_images"
OUTPUT_DIR = "."
TRAIN_DIR = os.path.join(OUTPUT_DIR, "coco_train")
TEST_DIR = os.path.join(OUTPUT_DIR, "coco_test")
TRAIN_JSON = os.path.join(OUTPUT_DIR, "annotations_train.json")
TEST_JSON = os.path.join(OUTPUT_DIR, "annotations_test.json")
SPLIT_RATIO = 0.8

# === СОЗДАНИЕ ПАПОК ===
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# === ЧТЕНИЕ CSV ===
df = pd.read_csv(CSV_PATH)

# === УНИКАЛЬНЫЕ ИЗОБРАЖЕНИЯ ===
all_images = df['image_name'].dropna().unique().tolist()
random.shuffle(all_images)
split_idx = int(len(all_images) * SPLIT_RATIO)
train_images = set(all_images[:split_idx])
test_images = set(all_images[split_idx:])

# === СОЗДАНИЕ СПИСКА КЛАССОВ ===
df['label'] = df['decoding'].astype(str).str.strip()
class_names = sorted(df['label'].unique())
class_id_map = {name: idx for idx, name in enumerate(class_names)}

# === ИНИЦИАЛИЗАЦИЯ COCO-СТРУКТУР ===
def init_coco():
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k} for k, v in class_id_map.items()]
    }

train_coco = init_coco()
test_coco = init_coco()

ann_id = 1
img_id_map = {}
image_cache = {}

# === ГЕНЕРАЦИЯ АННОТАЦИЙ ===
for idx, row in tqdm(df.iterrows(), total=len(df)):
    filename = row['image_name']
    if pd.isna(filename):
        continue

    # Получение размера изображения, если отсутствует в CSV
    if pd.isna(row['image_width']) or pd.isna(row['image_height']):
        if filename not in image_cache:
            image_path = os.path.join(IMG_DIR, filename)
            if not os.path.exists(image_path):
                continue
            with Image.open(image_path) as img:
                image_cache[filename] = img.size
        width, height = image_cache[filename]
    else:
        width = int(row['image_width'])
        height = int(row['image_height'])

    try:
        x_center = float(row['x_center']) * width
        y_center = float(row['y_center']) * height
        box_width = float(row['box_width']) * width
        box_height = float(row['box_height']) * height
    except:
        continue

    class_name = row['label']
    category_id = class_id_map.get(class_name, None)
    if category_id is None:
        continue

    x_min = x_center - box_width / 2
    y_min = y_center - box_height / 2

    # Создаём 4 угла прямоугольника (по часовой стрелке)
    x1, y1 = x_min, y_min
    x2, y2 = x_min + box_width, y_min
    x3, y3 = x_min + box_width, y_min + box_height
    x4, y4 = x_min, y_min + box_height
    segmentation = [[x1, y1, x2, y2, x3, y3, x4, y4]]

    # Добавляем изображение в COCO
    target = train_coco if filename in train_images else test_coco
    if filename not in img_id_map:
        img_id = len(img_id_map) + 1
        img_id_map[filename] = img_id
        coco_image = {
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
        }
        target["images"].append(coco_image)

        # Копирование изображения
        src = os.path.join(IMG_DIR, filename)
        dst = os.path.join(TRAIN_DIR if filename in train_images else TEST_DIR, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)

    # Добавляем аннотацию
    annotation = {
        "id": ann_id,
        "image_id": img_id_map[filename],
        "category_id": category_id,
        "bbox": [x_min, y_min, box_width, box_height],
        "area": box_width * box_height,
        "iscrowd": 0,
        "segmentation": segmentation
    }
    target["annotations"].append(annotation)
    ann_id += 1

# === СОХРАНЕНИЕ JSON ===
with open(TRAIN_JSON, "w", encoding='utf-8') as f:
    json.dump(train_coco, f, ensure_ascii=False, indent=2)

with open(TEST_JSON, "w", encoding='utf-8') as f:
    json.dump(test_coco, f, ensure_ascii=False, indent=2)

print("✅ Готово! Файлы сохранены с segmentation.")
