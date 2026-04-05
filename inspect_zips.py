import zipfile
import os

zips = [
    r"C:\Users\KANNAN-V\Downloads\archive.zip",
    r"C:\Users\KANNAN-V\Downloads\archive (1).zip",
    r"C:\Users\KANNAN-V\Downloads\archive (2).zip"
]

for zip_path in zips:
    print(f"\n--- Investigating: {os.path.basename(zip_path)} ---")
    if not os.path.exists(zip_path):
        print(f"File not found: {zip_path}")
        continue
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Get first 30 entries
        entries = z.namelist()
        
        # Check for YOLO format clues
        has_yaml = any(e.endswith('.yaml') for e in entries)
        has_labels = any('labels' in e for e in entries)
        has_annotations = any(e.endswith('.xml') or e.endswith('.json') for e in entries)
        
        print(f"Total files in zip: {len(entries)}")
        print(f"Has .yaml file: {has_yaml}")
        print(f"Has 'labels' folder: {has_labels}")
        print(f"Has XML/JSON annotations: {has_annotations}")
        
        # Print top-level folders / sample files
        top_levels = set([e.split('/')[0] for e in entries if '/' in e])
        print(f"Top-level directories: {list(top_levels)[:10]}")
        
        print("Sample files:")
        for idx in range(min(15, len(entries))):
             print(f"  {entries[idx]}")
