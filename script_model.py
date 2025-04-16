import os
import time
from PIL import Image
import torch

from dataset import AlignCollate  # ⚠️ Уже у тебя есть — используем

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 🔧 КОНФИГ ====
config = {
    "image_path": r"C:\A\a2\images",  # Путь до папки или одного изображения
    "torchscript_model_path": "frozen_model.pt",
    "imgH": 128,
    "imgW": 512,
    "input_channel": 1,
    "batch_max_length": 25,
    "rgb": False,
    "character": " !%'()*+,-./0123456789:;<=>[]^_v{|}~§«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёіѣѳѵ№",
}


# ==== 📦 ЗАГРУЗКА TorchScript-МОДЕЛИ ====
model = torch.jit.load(config["torchscript_model_path"], map_location=device)
model.eval()


# ==== 📷 ПРЕПРОЦЕССИНГ С ALIGNCOLLATE ====
def preprocess_image_aligncollate(image_path, config):
    img = Image.open(image_path).convert("RGB" if config["rgb"] else "L")
    align = AlignCollate(
        imgH=config["imgH"], imgW=config["imgW"], keep_ratio_with_pad=True
    )
    image_tensor, _ = align([(img, "")])
    return image_tensor.to(device)


# ==== 🧠 ПРЕДСКАЗАНИЕ ====
def predict(image_tensor, config):
    batch_size = image_tensor.size(0)
    text_input = torch.zeros(
        batch_size, config["batch_max_length"] + 1, dtype=torch.long
    ).to(device)

    with torch.no_grad():
        output = model(image_tensor, text_input)  # Attn

    _, pred_index = output.max(2)
    pred_index = pred_index.squeeze(0).tolist()

    # Расшифровка индексов в строку
    char_list = list(config["character"])
    char_list.insert(0, "[GO]")
    char_list.append("[s]")

    pred_str = ""
    for idx in pred_index:
        char = char_list[idx]
        if char == "[s]":
            break
        pred_str += char

    return pred_str


# ==== 🚀 ОСНОВНАЯ ФУНКЦИЯ ====
def run_inference(config):
    image_path = config["image_path"]
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
    print(f"\n🔍 Распознаём {len(image_files)} изображений...\n")

    for img_path in image_files:
        image_tensor = preprocess_image_aligncollate(img_path, config)

        start = time.time()
        pred = predict(image_tensor, config)
        end = time.time()

        duration = end - start
        total_time += duration

        print(
            f"{os.path.basename(img_path):<30} → {pred:<30} ({duration * 1000:.2f} ms)"
        )

    avg = total_time / len(image_files)
    print("\n" + "=" * 60)
    print(f"📊 Обработано: {len(image_files)} изображений")
    print(f"⏱ Среднее время инференса: {avg * 1000:.2f} ms")
    print("=" * 60)


# ==== 🔥 СТАРТ ====
if __name__ == "__main__":
    run_inference(config)
