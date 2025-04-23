import torch
from collections import OrderedDict
from model import Model

# Пример класса опций, как использовался при обучении
class Options:
    def __init__(self):
        self.imgH = 128
        self.imgW = 512
        self.num_fiducial = 20
        self.input_channel = 1     # grayscale
        self.output_channel = 512
        self.hidden_size = 256
        self.batch_max_length = 25
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        # Количество классов: набор символов + 2 (например, [GO] и [s])
        self.num_class = len(" !%'()*+,-./0123456789:;<=>[]^_v{|}~§«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёіѣѳѵ№") + 2

opt = Options()
model = Model(opt)

# Загружаем state_dict и убираем префикс "module." если он есть
model_path = r"shared/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_valid_loss.pth"
state_dict = torch.load(model_path, map_location='cpu')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)
model.eval()

# Подготовка фиктивных входов (dummy inputs) для трассировки.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
dummy_input = torch.randn(1, opt.input_channel, opt.imgH, opt.imgW).to(device)
dummy_text = torch.zeros(1, opt.batch_max_length + 1, dtype=torch.long).to(device)


# Используем torch.jit.trace для компиляции модели.
traced_model = torch.jit.trace(model, (dummy_input, dummy_text))

# Сохраняем скомпилированную модель
traced_model.save("model_traced.pt")
