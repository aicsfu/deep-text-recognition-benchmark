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


# ==== Функция для конвертации изображения в base64 HTML (для вставки в таблицу) ====
def image_to_base64_html(image_path, max_width=200):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    b64_data = base64.b64encode(img_data).decode("utf-8")
    return f'<img src="data:{mime_type};base64,{b64_data}" style="max-width: {max_width}px;"/>'


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
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        preds_str = [s.split("[s]")[0] for s in preds_str]
    recognized = preds_str[0]
    print("[DEBUG] Распознавание завершено")
    return recognized


# ==== Основной пайплайн для обработки полной страницы (с замером времени) ====
def process_full_image(image_path, opt, crops_dir):
    model, converter = load_model(opt)
    reader = easyocr.Reader(["ru", "en"], gpu=torch.cuda.is_available())
    print("[DEBUG] Считываем полное изображение через OpenCV...")
    image = cv2.imread(image_path)

    # Замер времени детекции (EasyOCR)
    start_detection = time.time()
    detections = reader.readtext(image)
    detection_time = time.time() - start_detection
    print(
        f"[DEBUG] Найдено {len(detections)} текстовых блоков. Время детекции: {detection_time:.2f} сек."
    )

    crop_results = []  # Для HTML-отчёта (thumbnail + prediction)
    annotations_list = []  # Для COCO-подобной разметки
    total_recognition_time = 0

    for i, (bbox, _, confidence) in enumerate(detections):
        pts = np.array(bbox).astype(int)
        x, y, w, h = cv2.boundingRect(pts)
        cropped_img = image[y : y + h, x : x + w]

        crop_path = os.path.join(crops_dir, f"word_{i:03}.png")
        cv2.imwrite(crop_path, cropped_img)
        print(f"[DEBUG] Сохранён crop {crop_path}")

        pil_crop = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        image_tensor_crop = preprocess_image_from_pil(pil_crop, opt)

        start_recognition = time.time()
        prediction = predict(model, converter, image_tensor_crop, opt)
        recognition_time = time.time() - start_recognition
        total_recognition_time += recognition_time

        # Добавляем данные для HTML-отчёта
        crop_results.append(
            {
                "thumbnail": image_to_base64_html(crop_path, max_width=150),
                "prediction": prediction,
            }
        )
        # Добавляем данные для COCO-подобной разметки (bbox в формате [x, y, w, h])
        annotation = {
            "id": i,
            "bbox": [int(x), int(y), int(w), int(h)],
            "text": prediction,
        }
        annotations_list.append(annotation)

        print(f"[{i:03}] → {prediction} ({recognition_time*1000:.2f} мс)")

    final_text = " ".join([item["prediction"] for item in crop_results])
    print("\n📜 Распознанный текст:")
    print(final_text)
    print(f"\n⏱ Время детекции: {detection_time:.2f} сек")
    print(
        f"⏱ Суммарное время распознавания блоков: {total_recognition_time*1000:.2f} мс"
    )

    # Генерация HTML-таблицы с обрезками и предсказаниями
    df_html = pd.DataFrame(crop_results, columns=["thumbnail", "prediction"])
    table_html = df_html.to_html(escape=False, index=False)

    # Оборачиваем таблицу в полноценный HTML-документ с указанием кодировки UTF-8
    full_html = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Отчёт OCR</title>
  </head>
  <body>
    {table_html}
  </body>
</html>
"""

    html_path = os.path.join(os.getcwd(), "results.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"[DEBUG] HTML-отчёт сохранён: {html_path}")

    # Генерация COCO-подобной разметки в виде JSON-словаря
    coco_annotations = {"annotations": annotations_list}
    json_path = os.path.join(os.getcwd(), "results.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(coco_annotations, json_file, ensure_ascii=False, indent=4)
    print(f"[DEBUG] COCO JSON разметка сохранена: {json_path}")

    return (
        final_text,
        crop_results,
        detection_time,
        total_recognition_time,
        html_path,
        json_path,
    )


# ==== Обработчик входящих фото и документов (файлов) в Telegram боте ====
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
