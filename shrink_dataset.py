import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import time

IMG_DIRS = ['dataset/train/images', 'dataset/valid/images']
MAX_SIZE = (640, 640)

def process_file(file_path):
    try:
        with Image.open(file_path) as img:
            # Check if it needs shrinking
            if img.width <= MAX_SIZE[0] and img.height <= MAX_SIZE[1]:
                return 0 # didn't shrink
            
            # Create a copy so we aren't mutating an open file handle heavily
            img_c = img.copy()
            
        # `.thumbnail` maintains aspect ratio!
        img_c.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to RGB so we can force JPEG compression
        if img_c.mode != 'RGB':
            img_c = img_c.convert('RGB')
            
        # Overwrite file with lower quality and smaller resolution
        img_c.save(file_path, 'JPEG', quality=85)
        return 1
    except Exception as e:
        return 0

def main():
    print("Gathering all image files...")
    files = []
    for d in IMG_DIRS:
        if os.path.exists(d):
            for f in os.listdir(d):
                ext = f.lower()
                if ext.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    files.append(os.path.join(d, f))
    
    total = len(files)
    print(f"Found {total} image files. Preparing to shrink to {MAX_SIZE[0]}x{MAX_SIZE[1]}...")
    
    start = time.time()
    processed = 0
    shrunk = 0
    
    # ProcessPoolExecutor massively speeds this up by using all CPU cores
    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_file, files, chunksize=250):
            processed += 1
            shrunk += result
            if processed % 1000 == 0:
                print(f"[{processed}/{total}] Resized {shrunk} images so far...")
                
    print(f"\nFinished in {time.time() - start:.2f} seconds!")
    print(f"Succesfully shrank {shrunk} extremely large images. Size should be vastly reduced!")
    
if __name__ == '__main__':
    main()
