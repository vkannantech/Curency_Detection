import zipfile
import os
import shutil
import yaml

zip_path = r'C:\Users\KANNAN-V\Downloads\indian currency-notes.v2i.yolov8.zip'
dataset_dir = r'D:\Code Space\Curency_Detection\dataset'

print(f"Extracting dataset from {zip_path}...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(dataset_dir)

yaml_path = os.path.join(dataset_dir, 'data.yaml')
print(f"Updating {yaml_path}")
if os.path.exists(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Fix paths for YOLO
    data['path'] = dataset_dir
    data['train'] = 'train/images'
    # Roboflow sometimes uses 'valid' instead of 'val'
    data['val'] = 'valid/images' if os.path.exists(os.path.join(dataset_dir, 'valid')) else 'val/images'
    data['test'] = 'test/images'
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    print("data.yaml updated successfully.")
    print("Classes in this dataset:")
    print(data.get('names', []))
else:
    print("data.yaml not found in the extracted files!")

print("Dataset setup complete!")
