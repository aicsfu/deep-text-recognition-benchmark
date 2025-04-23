import os
import time
from PIL import Image
import torch
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔧 НАСТРОЙКИ
image_path = r"C:\A\a2\images"  # ПАПКА с изображениями
saved_model = r"C:\best_accuracy.pth"

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
    img = Image.open(image_path).convert("RGB" if opt.rgb else "L")
    AlignCollate_infer = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
    )
    image_tensor, _ = AlignCollate_infer([(img, "")])
    return image_tensor.to(device)


def predict(model, converter, image, opt):
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


def infer(opt, image_path):
    model, converter = load_model(opt)

    # Получаем список изображений
    if os.path.isdir(image_path):
        image_files = sorted(
            [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
    else:
        image_files = [image_path]

    if not image_files:
        print("❌ Нет изображений для обработки.")
        return

    total_time = 0
    total_images = len(image_files)

    print(f"\n🔍 Обработка {total_images} изображений...\n")

    for img_path in image_files:
        image_tensor = preprocess_image(img_path, opt)

        start_time = time.time()
        prediction = predict(model, converter, image_tensor, opt)
        end_time = time.time()

        duration = end_time - start_time
        total_time += duration

        print(
            f"{os.path.basename(img_path):<30} → {prediction:<30} ({duration * 1000:.2f} ms)"
        )

    avg_time = total_time / total_images
    print("\n" + "=" * 60)
    print(f"📊 Обработано изображений: {total_images}")
    print(f"⏱ Среднее время инференса: {avg_time * 1000:.2f} ms")
    print("=" * 60)


# 🚀 Старт
if __name__ == "__main__":
    infer(opt, image_path)
