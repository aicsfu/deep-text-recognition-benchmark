import os
import shutil
import cv2
import numpy as np
from PIL import Image
import torch
import easyocr
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate

# ==== НАСТРОЙКИ ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = r"C:\A\a2\combined_images\1726.jpg"  # Путь к полной странице
saved_model = r"C:\best_accuracy.pth"

# Папка для обрезков
output_crop_dir = os.path.join(os.getcwd(), "crops")
if os.path.exists(output_crop_dir):
    shutil.rmtree(output_crop_dir)
os.makedirs(output_crop_dir, exist_ok=True)

# Параметры модели
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


# ==== Модель ====
def load_model(opt):
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    state_dict = torch.load(opt.saved_model, map_location=device)
    if "module" in list(state_dict.keys())[0]:
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict({"module." + k: v for k, v in state_dict.items()})
    model = model.module
    model.eval()
    return model, converter


# ==== Предобработка ====
def preprocess_image_from_pil(pil_img, opt):
    align = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    image_tensor, _ = align([(pil_img.convert("RGB" if opt.rgb else "L"), "")])
    return image_tensor.to(device)


# ==== Распознавание ====
def predict(model, converter, image_tensor, opt):
    batch_size = image_tensor.size(0)
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = (
        torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
    )

    with torch.no_grad():
        if "CTC" in opt.Prediction:
            preds = model(image_tensor, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
        else:
            preds = model(image_tensor, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            preds_str = [s.split("[s]")[0] for s in preds_str]

    return preds_str[0]


# ==== Основная обработка ====
def process_full_image(image_path, opt, output_crop_dir):
    model, converter = load_model(opt)
    reader = easyocr.Reader(["ru", "en"], gpu=torch.cuda.is_available())
    image = cv2.imread(image_path)
    detections = reader.readtext(image)

    print(f"\n🔍 Найдено {len(detections)} текстовых блоков...\n")

    predictions = []

    for i, (bbox, _, confidence) in enumerate(detections):
        if confidence < 0.0:
            continue

        pts = np.array(bbox).astype(int)
        x, y, w, h = cv2.boundingRect(pts)
        cropped_img = image[y : y + h, x : x + w]

        # Сохраняем каждое слово как обрезок
        crop_path = os.path.join(output_crop_dir, f"word_{i:03}.png")
        cv2.imwrite(crop_path, cropped_img)

        pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        image_tensor = preprocess_image_from_pil(pil_img, opt)
        prediction = predict(model, converter, image_tensor, opt)

        predictions.append(prediction)

    # Склеенный вывод
    final_text = " ".join(predictions)
    print("\n📜 Распознанный текст:")
    print(final_text)


# ==== СТАРТ ====
if __name__ == "__main__":
    process_full_image(image_path, opt, output_crop_dir)
