import os

def sync_images_and_labels(images_folder, labels_path):
    # 1. Загрузка данных
    with open(labels_path, 'r', encoding='utf-8') as f:
        label_lines = [line.strip() for line in f if line.strip()]

    label_dict = {}
    for line in label_lines:
        if '\t' in line:
            fname, text = line.split('\t', 1)
            label_dict[fname] = text

    label_filenames = set(label_dict.keys())
    image_filenames = set(f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

    # 2. Определение несоответствий
    labels_without_images = label_filenames - image_filenames
    images_without_labels = image_filenames - label_filenames

    # 3. Предварительный отчёт
    print("🔍 Предварительный анализ:")
    print(f"  Расшифровок без изображений: {len(labels_without_images)}")
    print(f"  Изображений без расшифровок: {len(images_without_labels)}")

    if not labels_without_images and not images_without_labels:
        print("✅ Всё уже синхронизировано. Нечего удалять.")
        return

    confirm = input("❓ Продолжить удаление этих элементов? (y/n): ").strip().lower()
    if confirm != 'y':
        print("🚫 Отменено пользователем.")
        return

    # 4. Удаление файлов
    for img in images_without_labels:
        os.remove(os.path.join(images_folder, img))
        print(f"🗑 Удалено изображение: {img}")

    # 5. Перезапись очищенного файла с расшифровками
    cleaned_labels = [f"{fname}\t{label_dict[fname]}" for fname in sorted(label_filenames - labels_without_images)]
    with open(labels_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_labels) + '\n')
    print(f"📝 Файл расшифровок обновлён: {labels_path}")

    print("✅ Синхронизация завершена.")

# Вызов:
sync_images_and_labels(
    images_folder=r"C:\shared\Archive_19_04\archive_sorted_all",  # <-- поменяй на свою папку с изображениями
    labels_path=r"C:\shared\Archive_19_04\gt_archive2.txt"
)
