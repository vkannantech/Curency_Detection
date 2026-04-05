import os
import zipfile
import shutil
import random
import uuid
import yaml

# Unified class map
# Original YOLO: 0:'10', 1:'100', 2:'20', 3:'200', 4:'2000', 5:'5', 6:'50', 7:'500', 8:'None'
# New Coins added: 9:'1', 10:'2'

CLASSES = {
    '10': 0,
    '100': 1,
    '20': 2,
    '200': 3,
    '2000': 4,
    '5': 5,
    '50': 6,
    '500': 7,
    'None': 8,
    'Background': 8, # maps to None
    '1': 9,
    '2': 10
}

WORD_MAP = {
    'One': '1',
    'Two': '2',
    'Five': '5',
    'Ten': '10',
    'Twenty': '20'
}

DATASET_ROOT = r'D:\Code Space\Curency_Detection\dataset'

def setup_dirs():
    for split in ['train', 'valid']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(DATASET_ROOT, split, sub), exist_ok=True)

def process_image(z, entry, class_str):
    if class_str not in CLASSES:
        return
    class_id = CLASSES[class_str]
    split = 'train' if random.random() < 0.8 else 'valid'
    
    # Extract file
    ext = os.path.splitext(entry)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
        return
        
    unique_name = uuid.uuid4().hex
    img_filename = f"{unique_name}{ext}"
    label_filename = f"{unique_name}.txt"
    
    img_dest = os.path.join(DATASET_ROOT, split, 'images', img_filename)
    label_dest = os.path.join(DATASET_ROOT, split, 'labels', label_filename)
    
    # Save image
    with z.open(entry) as source, open(img_dest, 'wb') as target:
        shutil.copyfileobj(source, target)
        
    # Save label (Center-Block pseudo label: 95% full frame)
    with open(label_dest, 'w') as f:
        f.write(f"{class_id} 0.5 0.5 0.95 0.95\n")

print("Setting up unified directories...")
setup_dirs()

# 1. Archive.zip
print("\nProcessing archive.zip...")
archive_path = r"C:\Users\KANNAN-V\Downloads\archive.zip"
if os.path.exists(archive_path):
    with zipfile.ZipFile(archive_path, 'r') as z:
        for entry in z.namelist():
            if '/' not in entry or entry.endswith('/'): continue
            parts = entry.split('/')
            cls = parts[-2]
            process_image(z, entry, cls)

# 2. Archive (1).zip
print("\nProcessing archive (1).zip...")
archive1_path = r"C:\Users\KANNAN-V\Downloads\archive (1).zip"
if os.path.exists(archive1_path):
    with zipfile.ZipFile(archive1_path, 'r') as z:
        for entry in z.namelist():
            if '/' not in entry or entry.endswith('/'): continue
            cls = entry.split('/')[-2]
            process_image(z, entry, cls)

# 3. Archive (2).zip
print("\nProcessing archive (2).zip...")
archive2_path = r"C:\Users\KANNAN-V\Downloads\archive (2).zip"
if os.path.exists(archive2_path):
    with zipfile.ZipFile(archive2_path, 'r') as z:
        for entry in z.namelist():
            if '/' not in entry or entry.endswith('/'): continue
            parts = entry.split('/')
            if len(parts) >= 3:
                word = parts[-4] if len(parts) > 3 else parts[-3] # 'DataSet/Five/Coin 1/Artificial Light/Five_1.jpg' -> parts[-4] is Five
                if word == 'DataSet' and len(parts) >= 3:
                     word = parts[-3] if len(parts) > 3 else parts[-2] # safety check
                # actually let's just search the path safely:
                found_word = None
                for p in parts:
                     if p in WORD_MAP:
                          found_word = p
                          break
                if found_word:
                    cls = WORD_MAP[found_word]
                    process_image(z, entry, cls)

print("\nUpdating data.yaml...")
yaml_path = os.path.join(DATASET_ROOT, 'data.yaml')
data = {}
if os.path.exists(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

data['path'] = DATASET_ROOT
data['train'] = 'train/images'
data['val'] = 'valid/images'
if 'test' in data:
    del data['test'] # test set might not be strictly organized now, YOLO can handle its absence
    
# Inverse lookup for names list
# ensure 'nc' and 'names' match the 11 classes perfectly in order 0-10
ordered_names = [None]*11
for k, v in CLASSES.items():
    if k == 'Background': continue
    ordered_names[v] = k

data['nc'] = 11
data['names'] = ordered_names

with open(yaml_path, 'w') as f:
    yaml.dump(data, f)

print("Unified dataset build successfully completed!")
