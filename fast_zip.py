import zipfile
import os
import time
import sys

folder_to_zip = 'dataset'
zip_filename = 'fast_dataset_upload.zip'

if os.path.exists(zip_filename):
    try:
        os.remove(zip_filename)
    except PermissionError:
        zip_filename = 'fast_dataset_upload_v2.zip'

start_time = time.time()

# The secret to speed: ZIP_STORED does not compress the files.
# Image files (.jpg, .png) are already heavily compressed natively. 
# Trying to re-compress them (what normal Zipping does) wastes 99% of your CPU time 
# and doesn't actually shrink the file size!

print(f"Creating {zip_filename} using ultra-fast STORE mode...")
total_files = sum([len(files) for r, d, files in os.walk(folder_to_zip)])
processed = 0

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_STORED, allowZip64=True) as zipf:
    for root, dirs, files in os.walk(folder_to_zip):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.join(folder_to_zip, os.path.relpath(file_path, folder_to_zip))
            zipf.write(file_path, arcname)
            processed += 1
            if processed % 1000 == 0:
                print(f"Packed {processed}/{total_files} files...")

duration = time.time() - start_time
size_mb = os.path.getsize(zip_filename) / (1024*1024)

print(f"\nDone! Packed {total_files} files in just {duration:.2f} seconds!")
print(f"Final file: {zip_filename} ({size_mb:.2f} MB)")
