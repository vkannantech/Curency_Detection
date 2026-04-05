import zipfile
import os

zips = {
    "archive.zip": r"C:\Users\KANNAN-V\Downloads\archive.zip",
    "archive_1.zip": r"C:\Users\KANNAN-V\Downloads\archive (1).zip",
    "archive_2.zip": r"C:\Users\KANNAN-V\Downloads\archive (2).zip"
}

for name, zip_path in zips.items():
    print(f"\n--- {name} ---")
    if not os.path.exists(zip_path):
        print("Missing file")
        continue
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = [f for f in z.namelist() if not f.endswith('/')]
        
        folders = set()
        prefixes = set()
        
        for f in files:
            parts = f.split('/')
            if len(parts) > 1:
                folders.add(parts[-2]) # Parent folder
            basename = parts[-1]
            if '_' in basename:
                prefixes.add(basename.split('_')[0])
                
        print(f"Parent folders: {list(folders)[:20]}")
        
        if name == "archive.zip":
            # For archive.zip, let's see unique prefixes since folders might be train/test
            print(f"File prefixes (first 20): {list(prefixes)[:20]}")
