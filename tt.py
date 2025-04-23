import pandas as pd

# Чтение исходного файла
df = pd.read_csv(r"C:\Users\USER\Desktop\generated_synth\generated_synth\test.csv")

# Создаём столбец sample_id (значения могут быть любыми, здесь просто сдвигаем индекс)
df['sample_id'] = df.index + 10

# Добавляем столбец stage со значением "train"
df['stage'] = "train"

# Переименовываем столбец filename в path
df.rename(columns={'filename': 'path'}, inplace=True)

# Упорядочиваем столбцы как требуется: sample_id, path, stage, text
df = df[['sample_id', 'path', 'stage', 'text']]

# Сохраняем в новый CSV файл
df.to_csv(r"C:\Users\USER\Desktop\generated_synth\generated_synth\marking.csv", index=False)