def merge_and_replace(file1_path, file2_path, output_path):
    with open(file1_path, 'r', encoding='utf-8') as f1, \
         open(file2_path, 'r', encoding='utf-8') as f2, \
         open(output_path, 'w', encoding='utf-8') as out:
        
        for line in f1:
            out.write(line)
        
        for line in f2:
            modified_line = line.replace("shared/result/images/", "test/")
            out.write(modified_line)

# Пример использования
merge_and_replace(r"C:\shared\result\gt.txt", r"C:\shared\reports15\reports15\data_stackmix\train\gt.txt", "merged_output.txt")