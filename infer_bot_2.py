import os
import shutil
import time
import cv2
import numpy as np
from PIL import Image
import torch
import easyocr
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
import nest_asyncio
import asyncio
import pandas as pd
import base64
import mimetypes
import json
from typing import List, Tuple
import math


# Применяем nest_asyncio, чтобы избежать ошибки "Cannot close a running event loop"
nest_asyncio.apply()

# === CONFIG ===
BOT_TOKEN = (
    "7287622548:AAGBEwjd5nhQS-XhGv4sa6Ihc06LOfZlHM4"  # Замените на свой реальный токен
)


# Модельный конфиг (из вашей системы)
class Opt:
    pass


opt = Opt()
opt.Transformation = "TPS"
opt.FeatureExtraction = "ResNet"
opt.SequenceModeling = "BiLSTM"
opt.Prediction = "Attn"
opt.character = " !%'()*+,-./0123456789:;<=>[]^_v{|}~§«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёіѣѳѵ№"
opt.batch_max_length = 25
opt.imgH = 128
opt.imgW = 512
opt.input_channel = 1  # Если rgb=False, то 1; иначе 3
opt.output_channel = 512
opt.hidden_size = 256
opt.rgb = False
opt.PAD = True
opt.num_fiducial = 20
opt.num_class = len(opt.character)
opt.sensitive = False
opt.data_filtering_off = True
opt.saved_model = r"C:\best_accuracy.pth"  # Путь к вашей модели

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Папка для сохранения обрезков (crops) — создаётся в текущей директории и очищается при каждом запуске
CROPS_DIR = os.path.join(os.getcwd(), "crops")
if os.path.exists(CROPS_DIR):
    shutil.rmtree(CROPS_DIR)
os.makedirs(CROPS_DIR, exist_ok=True)


import textwrap


def create_annotated_image_with_text(
    original_image_path: str,
    boxes: List[Tuple[int, int, int, int]],
    texts: List[str],
    confidences: List[float],  # Добавляем уверенности
    output_path: str,
    background_color: str = "white",
    text_color: str = "black",
    min_font_size: int = 6,
    max_font_size: int = 72,
    padding: int = 5,
    font_path: str = None,
    show_confidence: bool = False,  # Новая опция для отображения уверенности
) -> str:
    """
    Улучшенная версия: точно подбирает размер шрифта для идеального вписывания текста в бокс.

    Args:
        original_image_path: Путь к исходному изображению
        boxes: Список координатов боксов (x0, y0, x1, y1)
        texts: Список распознанных текстов для каждого бокса
        output_path: Куда сохранить результат
        background_color: Цвет фона
        text_color: Цвет текста
        min_font_size: Минимальный размер шрифта
        max_font_size: Максимальный размер шрифта
        padding: Отступ от границ бокса
        font_path: Путь к файлу шрифта (None - использует стандартный)
    """
    # Загружаем оригинальное изображение для получения размеров
    original_img = Image.open(original_image_path)
    img_width, img_height = original_img.size

    # Создаём новое изображение с указанным фоном
    annotated_img = Image.new("RGB", (img_width, img_height), color=background_color)
    draw = ImageDraw.Draw(annotated_img)

    for (x0, y0, x1, y1), text, confidence in zip(boxes, texts, confidences):
        if not text.strip():
            continue

        box_width = x1 - x0 - 2 * padding
        box_height = y1 - y0 - 2 * padding

        # Пропускаем слишком маленькие боксы
        if box_width <= 0 or box_height <= 0:
            continue

        # Функция для вычисления оптимального размера шрифта
        def calculate_font_size(font_size):
            try:
                font = (
                    ImageFont.truetype(font_path, font_size)
                    if font_path
                    else ImageFont.load_default(font_size)
                )
            except:
                font = ImageFont.load_default()

            # Получаем размеры текста
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            return font, text_width, text_height

        # Бинарный поиск оптимального размера шрифта
        low, high = min_font_size, max_font_size
        best_font = None
        best_size = min_font_size

        while low <= high:
            mid = (low + high) // 2
            font, text_width, text_height = calculate_font_size(mid)

            if text_width <= box_width and text_height <= box_height:
                best_font = font
                best_size = mid
                low = mid + 1  # Пробуем увеличить шрифт
            else:
                high = mid - 1  # Уменьшаем шрифт

        # Если не нашли подходящий шрифт, пробуем разбить на строки
        if best_font is None:
            font_size = min_font_size
            font, _, _ = calculate_font_size(font_size)

            # Рассчитываем примерное количество символов в строке
            avg_char_width = font_size * 0.6
            max_chars_per_line = max(1, int(box_width / avg_char_width))

            # Разбиваем текст на строки
            wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)
            text = "\n".join(wrapped_lines)

            # Проверяем, помещается ли многострочный текст
            lines_bbox = [
                draw.textbbox((0, 0), line, font=font) for line in wrapped_lines
            ]
            total_height = (
                sum(bbox[3] - bbox[1] for bbox in lines_bbox)
                + (len(wrapped_lines) - 1) * font_size * 0.2
            )

            if total_height <= box_height:
                best_font = font
            else:
                # Если всё равно не помещается, оставляем только часть текста
                max_lines = max(1, int(box_height / (font_size * 1.2)))
                text = "\n".join(wrapped_lines[:max_lines]) + (
                    "..." if len(wrapped_lines) > max_lines else ""
                )

        # Вычисляем позицию для текста (по центру бокса с учетом padding)
        text_bbox = draw.textbbox((0, 0), text, font=best_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        x = x0 + (x1 - x0 - text_width) / 2
        y = y0 + (y1 - y0 - text_height) / 2

        # Рисуем текст
        draw.text((x, y), text, fill=text_color, font=best_font)

        if show_confidence:
            conf_text = f"{confidence:.2f}"
            conf_bbox = draw.textbbox((0, 0), conf_text, font=best_font)
            conf_width = conf_bbox[2] - conf_bbox[0]
            conf_x = x1 - conf_width - 5
            conf_y = y1 - (conf_bbox[3] - conf_bbox[1]) - 5
            draw.text((conf_x, conf_y), conf_text, fill="blue", font=best_font)

    # Сохраняем результат
    annotated_img.save(output_path)
    return output_path


# ==== Функция для конвертации изображения в base64 HTML (для вставки в таблицу) ====
def image_to_base64_html(image_path, max_width=200):
    if not os.path.exists(image_path):
        return ""

    # Для аугментированных изображений делаем меньший размер
    width = max_width // 2 if "augmented" in image_path else max_width

    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    b64_data = base64.b64encode(img_data).decode("utf-8")
    return (
        f'<img src="data:{mime_type};base64,{b64_data}" style="max-width: {width}px;"/>'
    )


def apply_brightness_correction(image, target=100, max_gain=3.0):
    """Коррекция яркости с ограничением максимального усиления"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_mean = np.mean(gray)
    if current_mean < 1:  # избегаем деления на 0
        return image.copy()

    gain = min(target / current_mean, max_gain)
    result = cv2.convertScaleAbs(image, alpha=gain, beta=0)
    return result


def apply_clahe(image, clip_limit=4.0, tile_grid_size=(8, 8)):
    """Адаптивное выравнивание гистограммы"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_morphology(image, kernel_size=(3, 3), iterations=1):
    """Морфологические операции"""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def apply_color_augmentation(image, method: int):
    """
    Применяет цветовую аугментацию к изображению
    :param image: исходное изображение (numpy array)
    :param method: тип аугментации (0, 1, 2)
    :return: аугментированное изображение
    """
    if method == 0:
        # Яркостная коррекция (параметры для "фото")
        return apply_brightness_correction(image, target=170, max_gain=2.31)
    elif method == 1:
        # CLAHE с агрессивными параметрами
        return apply_clahe(image, clip_limit=4.0, tile_grid_size=(64, 64))
    elif method == 2:
        # Морфология + CLAHE
        img = apply_morphology(image, kernel_size=(2, 2), iterations=1)
        return apply_clahe(img, clip_limit=3.98, tile_grid_size=(16, 16))
    else:
        return image.copy()


# ==== Загрузка модели ====
def load_model(opt):
    from model import Model
    from utils import CTCLabelConverter, AttnLabelConverter

    print("[DEBUG] Загружаем модель...")
    # Для модели Attn используем AttnLabelConverter
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    state_dict = torch.load(opt.saved_model, map_location=device)
    print("[DEBUG] Загрузка state_dict завершена")
    if "module" in list(state_dict.keys())[0]:
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict({"module." + k: v for k, v in state_dict.items()})
    model = model.module
    model.eval()
    print("[DEBUG] Модель загружена и переведена в eval mode")
    return model, converter


# ==== Предобработка через AlignCollate (из PIL Image) ====
def preprocess_image_from_pil(pil_img, opt):
    from dataset import AlignCollate

    print("[DEBUG] Предобработка изображения через AlignCollate...")
    align = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    image_tensor, _ = align([(pil_img.convert("RGB" if opt.rgb else "L"), "")])
    print(f"[DEBUG] Тензор изображения: {image_tensor.shape}")
    return image_tensor.to(device)


# ==== Распознавание текста на блоке ====
def predict(model, converter, image_tensor, opt):
    print("[DEBUG] Запуск распознавания...")
    batch_size = image_tensor.size(0)
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = (
        torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
    )
    with torch.no_grad():
        preds = model(image_tensor, text_for_pred, is_train=False)

        # Вычисляем уверенность
        preds_prob = torch.nn.functional.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        preds_str = [s.split("[s]")[0] for s in preds_str]

        # Вычисляем среднюю уверенность по символам
        confidence = torch.mean(preds_max_prob).item()
    recognized = preds_str[0]
    print("[DEBUG] Распознавание завершено")
    return recognized, confidence


def predict_with_tta(model, converter, image_tensor, opt, original_image):
    """
    Распознавание с Test-Time Augmentation
    :return: список кортежей (текст, уверенность, тип аугментации)
    """
    predictions = []

    # Оригинальное изображение
    orig_pred, orig_conf = predict(model, converter, image_tensor, opt)
    predictions.append(("Original", orig_pred, orig_conf))

    # Создаем аугментированные версии
    for i in range(3):
        aug_img = apply_color_augmentation(original_image, i)
        aug_pil = Image.fromarray(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
        aug_tensor = preprocess_image_from_pil(aug_pil, opt)
        pred, conf = predict(model, converter, aug_tensor, opt)
        predictions.append((f"Augmentation {i+1}", pred, conf))

    return predictions


from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def segment_columns(
    boxes: List[Tuple[int, int, int, int]], img_width: int, max_splits: int = None
) -> List[List[Tuple[int, int, int, int]]]:
    def find_gaps(start: int, end: int) -> List[int]:
        segs = []
        for x0, _, x1, _ in boxes:
            if x1 <= start or x0 >= end:
                continue
            segs.append((max(x0, start), min(x1, end)))
        if not segs:
            return []
        segs.sort()
        merged = [segs[0]]
        for s, e in segs[1:]:
            ms, me = merged[-1]
            if s <= me:
                merged[-1] = (ms, max(me, e))
            else:
                merged.append((s, e))
        gaps = []
        prev = start
        for s, e in merged:
            if s > prev:
                gaps.append((prev, s))
            prev = max(prev, e)
        if prev < end:
            gaps.append((prev, end))
        return [(a + b) // 2 for a, b in gaps if b - a > 1]

    def emptiness(start: int, end: int) -> float:
        col = [b for b in boxes if b[0] >= start and b[2] <= end]
        if not col:
            return 1.0
        min_y = min(b[1] for b in col)
        max_y = max(b[3] for b in col)
        rect_area = (end - start) * (max_y - min_y)
        boxes_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in col)
        return (rect_area - boxes_area) / rect_area

    segments: List[Tuple[int, int]] = [(0, img_width)]
    separators: List[int] = []
    for _ in range(max_splits or img_width):
        best = None
        for idx, (s, e) in enumerate(segments):
            for x in find_gaps(s, e):
                left = any(b[2] <= x and b[0] >= s for b in boxes)
                right = any(b[0] >= x and b[2] <= e for b in boxes)
                if not (left and right):
                    continue
                ce = emptiness(s, x) + emptiness(x, e)
                if best is None or ce < best[0]:
                    best = (ce, x, idx)
        if not best:
            break
        _, x_split, idx = best
        s, e = segments.pop(idx)
        separators.append(x_split)
        segments.insert(idx, (s, x_split))
        segments.insert(idx + 1, (x_split, e))
        segments.sort()

    cols: List[List[Tuple[int, int, int, int]]] = []
    segs = [(0, img_width)]
    for x in separators:
        new_segs = []
        for s, e in segs:
            if s < x < e:
                new_segs.extend([(s, x), (x, e)])
            else:
                new_segs.append((s, e))
        segs = new_segs
    for s, e in segs:
        cols.append([b for b in boxes if b[0] >= s and b[2] <= e])
    cols_sorted = sorted(
        cols, key=lambda col: min(b[0] for b in col) if col else img_width
    )
    return cols_sorted


def sort_boxes_reading_order(boxes, y_tol_ratio=0.6, x_gap_ratio=2.5):
    heights = [y1 - y0 for x0, y0, x1, y1 in boxes]
    avg_height = np.mean(heights)

    def box_center_y(b):
        return (b[1] + b[3]) / 2

    boxes = sorted(boxes, key=box_center_y)

    lines = []
    for box in boxes:
        x0, y0, x1, y1 = box
        cy = (y0 + y1) / 2
        placed = False

        for line in lines:
            line_cy = np.mean([(b[1] + b[3]) / 2 for b in line])
            if abs(cy - line_cy) <= avg_height * y_tol_ratio:
                max_x1 = max(b[2] for b in line)
                if x0 - max_x1 < avg_height * x_gap_ratio:
                    line.append(box)
                    placed = True
                    break

        if not placed:
            lines.append([box])

    lines.sort(key=lambda line: np.mean([(b[1] + b[3]) / 2 for b in line]))
    for line in lines:
        line.sort(key=lambda b: b[0])

    return [box for line in lines for box in line]


def draw_boxes_on_image(
    image: Image.Image, boxes: List[Tuple[int, int, int, int]], font_size: int = 20
) -> Image.Image:
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for idx, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, y0 - font_size), str(idx), font=font, fill="blue")

    return img


from PIL import Image
import easyocr


# Обработка изображения: распознавание, сегментация, сортировка, визуализация
def process_image_with_columns(
    image_path: str, scale: float = 1.0, font_size: int = 20, max_splits: int = 3
):
    image = Image.open(image_path)
    w, h = image.size
    new_w, new_h = int(w * scale), int(h * scale)
    img_scaled = image.resize((new_w, new_h), resample=Image.LANCZOS)

    # Распознавание
    reader = easyocr.Reader(["en", "ru"], gpu=True)
    arr = np.array(image)
    results = reader.readtext(arr, detail=1, x_ths=0.0, width_ths=0.0, y_ths=0.0)

    # Преобразование боксов и масштабирование
    raw_boxes = []
    for res in results:
        bbox = res[0]
        xs = [int(pt[0]) for pt in bbox]
        ys = [int(pt[1]) for pt in bbox]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        raw_boxes.append(
            (int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale))
        )

    # Сегментация по колонкам
    columns = segment_columns(raw_boxes, img_width=new_w, max_splits=max_splits)

    # Сортировка внутри каждой колонки
    sorted_all = []
    for col in columns:
        sorted_col = sort_boxes_reading_order(col)
        sorted_all.extend(sorted_col)

    # Визуализация
    annotated_image = draw_boxes_on_image(img_scaled, sorted_all, font_size=font_size)
    return annotated_image, sorted_all


def save_augmented_crops(
    crop_img: np.ndarray, crop_dir: str, crop_name: str
) -> List[Tuple[str, str]]:
    """
    Сохраняет аугментированные версии обрезка и возвращает пути с описанием
    """
    aug_dir = os.path.join(crop_dir, "augmented")
    os.makedirs(aug_dir, exist_ok=True)

    augmented = []

    # Оригинал
    orig_path = os.path.join(aug_dir, f"{crop_name}_orig.png")
    cv2.imwrite(orig_path, crop_img)
    augmented.append(("Original", orig_path))

    # Аугментации
    aug1 = apply_color_augmentation(crop_img, 0)
    path1 = os.path.join(aug_dir, f"{crop_name}_aug1.png")
    cv2.imwrite(path1, aug1)
    augmented.append(("Augmentation 1", path1))  # Вместо "Brightness Correction"

    aug2 = apply_color_augmentation(crop_img, 1)
    path2 = os.path.join(aug_dir, f"{crop_name}_aug2.png")
    cv2.imwrite(path2, aug2)
    augmented.append(("Augmentation 2", path2))  # Вместо "CLAHE"

    aug3 = apply_color_augmentation(crop_img, 2)
    path3 = os.path.join(aug_dir, f"{crop_name}_aug3.png")
    cv2.imwrite(path3, aug3)
    augmented.append(("Augmentation 3", path3))  # Вместо "Morphology + CLAHE"

    return augmented


# ==== Основной пайплайн для обработки полной страницы (с замером времени) ====
def process_full_image(image_path, opt, crops_dir):
    from PIL import Image

    model, converter = load_model(opt)

    print("[DEBUG] Считываем изображение (OpenCV + PIL)...")
    image = cv2.imread(image_path)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    w, h = image_pil.size

    # === Новый порядок: упорядоченные боксы через сегментацию и сортировку ===
    start_detection = time.time()
    annotated_img, ordered_boxes = process_image_with_columns(
        image_path=image_path,
        scale=1.0,
        font_size=20,
        max_splits=3,
    )
    detection_time = time.time() - start_detection
    print(
        f"[DEBUG] Найдено {len(ordered_boxes)} текстовых блоков. Время детекции: {detection_time:.2f} сек."
    )

    crop_results = []
    annotations_list = []
    total_recognition_time = 0
    recognized_texts = []  # Список распознанных текстов для каждого бокса
    confidences = []  # Новый список для хранения уверенностей

    # В основном цикле обработки блоков:
    for i, (x0, y0, x1, y1) in enumerate(ordered_boxes):
        cropped_img = image[y0:y1, x0:x1]
        crop_path = os.path.join(crops_dir, f"word_{i:03}.png")
        cv2.imwrite(crop_path, cropped_img)

        # Сохраняем аугментированные версии и получаем пути
        augmented_data = save_augmented_crops(cropped_img, crops_dir, f"word_{i:03}")

        # Получаем предсказания для всех аугментаций
        pil_crop = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        image_tensor_crop = preprocess_image_from_pil(pil_crop, opt)

        # Измеряем время распознавания для этого блока
        start_recognition = time.time()
        tta_results = predict_with_tta(
            model, converter, image_tensor_crop, opt, cropped_img
        )
        recognition_time = time.time() - start_recognition
        total_recognition_time += recognition_time

        # Формируем данные для отчета
        augments = []
        for aug_type, pred, conf in tta_results:
            img_path = next(
                (
                    path
                    for desc, path in augmented_data
                    if desc.lower() in aug_type.lower()
                ),
                "",
            )
            augments.append(
                {
                    "type": aug_type,
                    "prediction": pred,  # Добавляем распознанный текст
                    "confidence": f"{conf:.4f}",
                    "image": image_to_base64_html(img_path) if img_path else "",
                }
            )

        # Находим лучший результат
        best_result = max(tta_results, key=lambda x: x[2])
        prediction = best_result[1]
        confidence = best_result[2]

        # Сохраняем данные для аннотаций
        recognized_texts.append(prediction)
        confidences.append(confidence)

        crop_results.append(
            {
                "thumbnail": image_to_base64_html(crop_path),
                "prediction": prediction,
                "confidence": f"{confidence:.4f}",
                "augmentations": augments,
            }
        )

        annotation = {
            "id": i,
            "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
            "text": prediction,
            "confidence": confidence,
        }
        annotations_list.append(annotation)

        print(
            f"[{i:03}] → {prediction} (conf: {confidence:.4f}) ({recognition_time*1000:.2f} мс)"
        )

    final_text = " ".join([item["prediction"] for item in crop_results])
    print("\n📜 Распознанный текст:")
    print(final_text)
    print(f"\n⏱ Время детекции: {detection_time:.2f} сек")
    print(
        f"⏱ Суммарное время распознавания блоков: {total_recognition_time*1000:.2f} мс"
    )

    # HTML
    df_html = pd.DataFrame(
        crop_results, columns=["thumbnail", "prediction", "confidence", "augmentations"]
    )
    table_html = df_html.to_html(
        escape=False,
        index=False,
        formatters={
            "thumbnail": lambda x: x,
            "augmentations": lambda augs: "<br>".join(
                [
                    f"""<div style="border:1px solid #ccc; margin:5px; padding:5px;">
                {a['image']}
                <div style="font-size:12px;">
                    Тип: {a['type']}<br>
                    Текст: {a['prediction']}<br>  <!-- Прямой доступ к ключу -->
                    Confidence: {a['confidence']}
                </div>
            </div>"""
                    for a in augs
                ]
            ),
        },
    )
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Отчёт OCR</title></head><body>{table_html}</body></html>
"""
    html_path = os.path.join(os.getcwd(), "results.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"[DEBUG] HTML-отчёт сохранён: {html_path}")

    # JSON
    coco_annotations = {"annotations": annotations_list}
    json_path = os.path.join(os.getcwd(), "results.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(coco_annotations, json_file, ensure_ascii=False, indent=4)
    print(f"[DEBUG] JSON разметка сохранена: {json_path}")

    # Создаём изображение с распознанным текстом на белом фоне
    # Создаём изображение с распознанным текстом на белом фоне
    text_image_path = os.path.join(os.getcwd(), "recognized_text.png")
    create_annotated_image_with_text(
        original_image_path=image_path,
        boxes=ordered_boxes,
        texts=recognized_texts,
        output_path=text_image_path,
        confidences=confidences,
        background_color="white",
        text_color="black",
        min_font_size=6,
        max_font_size=72,
        padding=5,
        font_path="arial.ttf",
        show_confidence=True,  # Добавлен параметр
    )
    print(f"[DEBUG] Изображение с распознанным текстом сохранено: {text_image_path}")

    return (
        final_text,
        crop_results,
        detection_time,
        total_recognition_time,
        html_path,
        json_path,
        text_image_path,  # Добавляем путь к новому изображению в возвращаемые значения
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("[DEBUG] Получено сообщение с фото")
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    local_path = os.path.join(os.getcwd(), "received.jpg")
    print(f"[DEBUG] Скачивание фото в {local_path}...")
    await file.download_to_drive(local_path)
    print("[DEBUG] Фото скачано")

    start = time.time()
    (
        final_text,
        crop_results,
        detection_time,
        total_recognition_time,
        html_path,
        json_path,
        text_image_path,  # Новый возвращаемый параметр
    ) = process_full_image(local_path, opt, CROPS_DIR)
    total_duration = time.time() - start

    reply = (
        f"📜 Распознанный текст:\n{final_text}\n\n"
        f"⏱ Время детекции: {detection_time:.2f} сек\n"
        f"⏱ Суммарное время распознавания: {total_recognition_time*1000:.2f} мс\n"
        f"⏱ Общее время обработки: {total_duration:.2f} сек"
    )
    print(f"[DEBUG] Ответ: {reply}")
    await update.message.reply_text(reply)

    # Отправляем изображение с распознанным текстом
    with open(text_image_path, "rb") as img_file:
        await update.message.reply_photo(photo=img_file, caption="Распознанный текст")
    print("[DEBUG] Изображение с распознанным текстом отправлено пользователю.")

    # Отправляем HTML-отчёт как документ
    with open(html_path, "rb") as html_file:
        await update.message.reply_document(document=html_file, filename="results.html")
    print("[DEBUG] HTML-отчёт отправлен пользователю.")

    # Отправляем JSON с COCO-разметкой как документ
    with open(json_path, "rb") as json_file:
        await update.message.reply_document(document=json_file, filename="results.json")
    print("[DEBUG] JSON разметка отправлена пользователю.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    if not document.mime_type.startswith("image/"):
        await update.message.reply_text(
            "Пожалуйста, отправьте документ с изображением."
        )
        return

    print("[DEBUG] Получен документ с изображением")
    file = await context.bot.get_file(document.file_id)
    local_path = os.path.join(os.getcwd(), "received.jpg")
    print(f"[DEBUG] Скачивание документа в {local_path}...")
    await file.download_to_drive(local_path)
    print("[DEBUG] Документ скачан")

    start = time.time()
    (
        final_text,
        crop_results,
        detection_time,
        total_recognition_time,
        html_path,
        json_path,
        text_image_path,  # Новый возвращаемый параметр
    ) = process_full_image(local_path, opt, CROPS_DIR)
    total_duration = time.time() - start

    reply = (
        f"📜 Распознанный текст:\n{final_text}\n\n"
        f"⏱ Время детекции: {detection_time:.2f} сек\n"
        f"⏱ Суммарное время распознавания: {total_recognition_time*1000:.2f} мс\n"
        f"⏱ Общее время обработки: {total_duration:.2f} сек"
    )
    print(f"[DEBUG] Ответ: {reply}")
    await update.message.reply_text(reply)

    # Отправляем изображение с распознанным текстом
    with open(text_image_path, "rb") as img_file:
        await update.message.reply_photo(photo=img_file, caption="Распознанный текст")
    print("[DEBUG] Изображение с распознанным текстом отправлено пользователю.")

    # Отправляем HTML-отчёт как документ
    with open(html_path, "rb") as html_file:
        await update.message.reply_document(document=html_file, filename="results.html")
    print("[DEBUG] HTML-отчёт отправлен пользователю.")

    # Отправляем JSON с COCO-разметкой как документ
    with open(json_path, "rb") as json_file:
        await update.message.reply_document(document=json_file, filename="results.json")
    print("[DEBUG] JSON разметка отправлена пользователю.")


# ==== Запуск бота ====
async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    print("[DEBUG] Бот запущен, ожидаем сообщений...")
    await app.run_polling(close_loop=False)


if __name__ == "__main__":
    asyncio.run(main())
