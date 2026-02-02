from pathlib import Path

def folder_size(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())

def human_readable(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

folder1 = Path("")
folder2 = Path("")

size1 = folder_size(folder1)
size2 = folder_size(folder2)

if size2 != 0:
    percent_diff = abs(size1 - size2) / size2 * 100
else:
    percent_diff = 0

print("Folder Size Comparison\n")

print(f"Folder 1 (WSL): {folder1}")
print(f"Size: {human_readable(size1)}\n")

print(f"Folder 2 (Windows): {folder2}")
print(f"Size: {human_readable(size2)}\n")

print(f"Percentage difference: {percent_diff:.2f}%")
