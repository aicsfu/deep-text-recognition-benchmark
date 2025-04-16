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

    predictions = []
    total_recognition_time = 0
    recognized_count = 0

    for i, (bbox, _, confidence) in enumerate(detections):
        # Здесь можно настроить порог уверенности (оставляем все блоки, confidence >= 0.0)
        if confidence < 0.0:
            print(f"[DEBUG] Блок {i:03} пропущен (confidence {confidence:.2f})")
            continue

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
        recognized_count += 1

        predictions.append(prediction)
        print(f"[{i:03}] → {prediction} ({recognition_time*1000:.2f} мс)")

    final_text = " ".join(predictions)
    print("\n📜 Распознанный текст:")
    print(final_text)
    print(f"\n⏱ Время детекции: {detection_time:.2f} сек")
    print(
        f"⏱ Суммарное время распознавания блоков: {total_recognition_time*1000:.2f} мс"
    )
    return final_text, predictions, detection_time, total_recognition_time


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
    final_text, _, detection_time, total_recognition_time = process_full_image(
        local_path, opt, CROPS_DIR
    )
    total_duration = time.time() - start

    reply = (
        f"📜 Распознанный текст:\n{final_text}\n\n"
        f"⏱ Время детекции: {detection_time:.2f} сек\n"
        f"⏱ Суммарное время распознавания: {total_recognition_time*1000:.2f} мс\n"
        f"⏱ Общее время обработки: {total_duration:.2f} сек"
    )
    print(f"[DEBUG] Ответ: {reply}")
    await update.message.reply_text(reply)


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
    final_text, _, detection_time, total_recognition_time = process_full_image(
        local_path, opt, CROPS_DIR
    )
    total_duration = time.time() - start

    reply = (
        f"📜 Распознанный текст:\n{final_text}\n\n"
        f"⏱ Время детекции: {detection_time:.2f} сек\n"
        f"⏱ Суммарное время распознавания: {total_recognition_time*1000:.2f} мс\n"
        f"⏱ Общее время обработки: {total_duration:.2f} сек"
    )
    print(f"[DEBUG] Ответ: {reply}")
    await update.message.reply_text(reply)


# ==== Запуск бота ====
async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    print("[DEBUG] Бот запущен, ожидаем сообщений...")
    await app.run_polling(close_loop=False)


if __name__ == "__main__":
    asyncio.run(main())
