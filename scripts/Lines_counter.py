import os

def count_py_lines(directory):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
    return total_lines

directory_path = r'E:\Pothole-tracking-comprehensive\scripts'  # Change this
print(f"Total .py lines: {count_py_lines(directory_path)}")