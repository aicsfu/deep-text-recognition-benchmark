import os
import time
import base64
import mimetypes
import pandas as pd
from PIL import Image
import torch
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate

# Выбор устройства (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔧 НАСТРОЙКИ
base_dir = r"C:\A\a2"
image_folder = os.path.join(base_dir, "images")
saved_model = r"C:\best_accuracy.pth"
csv_path = os.path.join(base_dir, "marking.csv")  # CSV с разметкой

Transformation = "TPS"
FeatureExtraction = "ResNet"
SequenceModeling = "BiLSTM"
Prediction = "Attn"

character = " !%'()*+,-./0123456789:;<=>[]^_v{|}~§«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёіѣѳѵ№"
batch_max_length = 25
imgH = 128
imgW = 512
input_channel = 1
output_channel = 512
hidden_size = 256
rgb = False
PAD = True


# ==== КОНФИГ ====
class Opt:
    pass


opt = Opt()
opt.Transformation = Transformation
opt.FeatureExtraction = FeatureExtraction
opt.SequenceModeling = SequenceModeling
opt.Prediction = Prediction
opt.character = character
opt.batch_max_length = batch_max_length
opt.imgH = imgH
opt.imgW = imgW
opt.input_channel = 3 if rgb else 1
opt.output_channel = output_channel
opt.hidden_size = hidden_size
opt.saved_model = saved_model
opt.rgb = rgb
opt.PAD = PAD
opt.num_fiducial = 20
opt.num_class = len(character)
opt.sensitive = False
opt.data_filtering_off = True


def load_model(opt):
    # Создаем конвертер для модели
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    print(f"Загрузка модели из {opt.saved_model}")
    state_dict = torch.load(opt.saved_model, map_location=device)
    if "module" in list(state_dict.keys())[0]:
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict({"module." + k: v for k, v in state_dict.items()})

    model = model.module
    model.eval()
    return model, converter


def preprocess_image(image_path, opt):
    """Открываем изображение, преобразуем в нужный формат и возвращаем тензор"""
    img = Image.open(image_path).convert("RGB" if opt.rgb else "L")
    AlignCollate_infer = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
    )
    image_tensor, _ = AlignCollate_infer([(img, "")])
    return image_tensor.to(device)


def predict(model, converter, image, opt):
    """Получаем предсказание модели для одного изображения"""
    batch_size = image.size(0)
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = (
        torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
    )

    with torch.no_grad():
        if "CTC" in opt.Prediction:
            preds = model(image, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
        else:
            preds = model(image, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            preds_str = [s.split("[s]")[0] for s in preds_str]

    return preds_str[0]


def image_to_base64_html(image_path, max_width=200):
    """Преобразует изображение в строку HTML с тегом <img> (встроенная картинка через base64)."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    b64_data = base64.b64encode(img_data).decode("utf-8")
    return f'<img src="data:{mime_type};base64,{b64_data}" style="max-width: {max_width}px;"/>'


def process_csv_predictions(opt, csv_path, base_dir):
    # Загружаем CSV с разметкой
    df_mark = pd.read_csv(csv_path).iloc[:1000]
    results = []

    # Загружаем модель один раз
    model, converter = load_model(opt)

    total = len(df_mark)
    print(f"\n🔍 Обработка {total} изображений согласно CSV...\n")
    for idx, row in df_mark.iterrows():
        # Формирование полного пути к изображению
        rel_path = row["path"].replace("/", os.sep).replace("\\", os.sep)
        image_path = (
            os.path.join(base_dir, rel_path)
            if not os.path.isabs(rel_path)
            else rel_path
        )

        # Проверяем существование файла
        if not os.path.exists(image_path):
            print(f"❌ Файл не найден: {image_path}")
            pred = ""
            thumb = ""
        else:
            image_tensor = preprocess_image(image_path, opt)
            start_time = time.time()
            pred = predict(model, converter, image_tensor, opt)
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            print(
                f"{os.path.basename(image_path):<30} → {pred:<30} ({duration:.2f} ms)"
            )
            # Создаем HTML-миниатюру изображения
            thumb = image_to_base64_html(image_path, max_width=100)

        # Сравнение предсказания с истинным текстом (убираем лишние пробелы и приводим к одному регистру)
        truth = str(row["text"]).strip()
        prediction = pred.strip() if pred else ""
        # Если необходимо учитывать регистр, удалите .lower()
        correct = truth.lower() == prediction.lower()

        results.append(
            {
                "correct": correct,
                "path": image_path,
                "truth": truth,
                "prediction": prediction,
                "thumbnail": thumb,
            }
        )

    # Создаем DataFrame с результатами
    df_results = pd.DataFrame(results)
    # Сохраняем в CSV (как и раньше)
    results_csv = os.path.join(base_dir, "results.csv")
    df_results.to_csv(results_csv, index=False, encoding="utf-8-sig")
    print(f"\n📊 Результаты сохранены в CSV: {results_csv}")

    # Для HTML-таблицы используем миниатюру вместо пути
    # Упорядочим столбцы по: миниатюра, истина, предсказание, корректность
    df_html = df_results[["thumbnail", "truth", "prediction", "correct"]]
    results_html = os.path.join(base_dir, "results.html")
    with open(results_html, "w", encoding="utf-8") as f:
        f.write(df_html.to_html(escape=False, index=False))
    print(f"🌐 Результаты сохранены в HTML: {results_html}")


# 🚀 Старт обработки
if __name__ == "__main__":
    process_csv_predictions(opt, csv_path, base_dir)
