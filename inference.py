import cv2
import torch
import torch.nn.functional as F
import numpy as np
import re

# Импортируем конвертеры меток (учтите, что они должны соответствовать используемому Prediction)
from utils import CTCLabelConverter, AttnLabelConverter

def inference(model, cv_image):
    """
    Функция для инференса. Принимает:
      - model: уже инициализированная (и загруженная) модель
      - cv_image: изображение в формате cv2 (numpy-массив)
    
    Внутри функции задаются все параметры модели.
    Возвращает предсказанный текст.
    """
    # Определяем устройство (GPU если доступно)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # ==================== Параметры инференса ====================
    imgH = 32                   # высота изображения (должна совпадать с параметром обучения)
    imgW = 100                  # ширина изображения (должна совпадать с параметром обучения)
    batch_max_length = 25       # максимальная длина строки
    rgb = False                 # использовать ли RGB (если False, то ожидается grayscale)
    Prediction = 'CTC'          # тип предсказания: 'CTC' или 'Attn'
    # Набор символов, используемый при обучении (должен совпадать)
    character = " !%'()*+,-./0123456789:;<=>[]^_v{|}~§«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёіѣѳѵ№"
    # ============================================================

    # -------------------- Предобработка изображения --------------------
    # Если ожидается одноканальное изображение (grayscale), то преобразуем:
    if not rgb:
        if len(cv_image.shape) == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        # Если rgb, преобразуем из BGR (формат cv2) в RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Изменяем размер изображения до фиксированных (imgW x imgH)
    # Здесь можно добавить сохранение пропорций с паддингом, если требуется, но для простоты используется прямое изменение размера.
    resized_image = cv2.resize(cv_image, (imgW, imgH))
    
    # Нормализуем изображение (приводим значения пикселей к диапазону [0, 1])
    resized_image = resized_image.astype(np.float32) / 255.0

    # Если изображение одноканальное, добавляем размер канала
    if not rgb:
        resized_image = np.expand_dims(resized_image, axis=2)
    
    # Транспонируем изображение в формат (C, H, W)
    resized_image = resized_image.transpose(2, 0, 1)
    
    # Преобразуем в тензор и добавляем размер батча
    image_tensor = torch.from_numpy(resized_image).unsqueeze(0).to(device)
    # --------------------------------------------------------------------

    # -------------------- Создаем конвертер меток --------------------
    if 'CTC' in Prediction:
        converter = CTCLabelConverter(character)
    else:
        converter = AttnLabelConverter(character)
    # -----------------------------------------------------------------

    # -------------------- Прямой проход через модель --------------------
    if 'CTC' in Prediction:
        # Для CTC требуется создать тензор для "текста предсказания"
        text_for_pred = torch.LongTensor(1, batch_max_length + 1).fill_(0).to(device)
        preds = model(image_tensor, text_for_pred)
        preds_size = torch.IntTensor([preds.size(1)]).to(device)
        # Получаем индексы с максимальной вероятностью на каждом временном шаге
        _, preds_index = preds.max(2)
        # Декодируем последовательность в строку
        preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    else:
        # Для Attention-based модели
        length_for_pred = torch.IntTensor([batch_max_length]).to(device)
        text_for_pred = torch.LongTensor(1, batch_max_length + 1).fill_(0).to(device)
        preds = model(image_tensor, text_for_pred, is_train=False)
        # Для Attn-модели обрезаем [GO] токен
        preds = preds[:, :batch_max_length, :]
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)[0]
    # -----------------------------------------------------------------

    return preds_str

# ==================== Пример использования ====================
if __name__ == '__main__':
    # Импорт модели (убедитесь, что структура модели совпадает с обученной)
    from model import Model

    # Пример: создаём класс с опциями для инициализации модели (смотрите train скрипт)
    class Options:
        def __init__(self):
            self.imgH = 512
            self.imgW = 128
            self.num_fiducial = 20
            self.input_channel = 1     # или 3, если RGB
            self.output_channel = 512
            self.hidden_size = 256
            self.batch_max_length = 25
            self.Transformation = 'TPS'      # или 'TPS'
            self.FeatureExtraction = 'ResNet'   # например, 'ResNet'
            self.SequenceModeling = 'BiLSTM'    # например, 'BiLSTM'
            self.Prediction = 'Attn'             # или 'Attn'
            # Количество классов должно совпадать с длиной строки символов (с учётом [blank] для CTC)
            self.num_class = len(" !%'()*+,-./0123456789:;<=>[]^_v{|}~§«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёіѣѳѵ№")
    
    opt = Options()
    # Инициализируем модель
    model = Model(opt)
    # Если имеется файл с предобученными весами, загрузите их:
    model.load_state_dict(torch.load(r'C:\shared\saved_models\TPS-ResNet-BiLSTM-Attn-Seed1111\best_valid_loss.pth', map_location='cpu'))
    model.eval()

    # Загрузите изображение для инференса (путь к изображению)
    img_path = r'demo_image\demo_8.jpg'
    cv_img = cv2.imread(img_path)
    if cv_img is None:
        print(f"Ошибка загрузки изображения: {img_path}")
    else:
        predicted_text = inference(model, cv_img)
        print("Распознанный текст:", predicted_text)
