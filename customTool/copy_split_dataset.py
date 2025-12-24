import os
import shutil

def main():
    source_root = 'data/guilinflood'
    target_root = 'data/guilinflood_split'
    split_dir = os.path.join(source_root, 'splits')
    
    # Define mappings
    # val.txt maps to 'test' folder as requested by user
    split_mapping = {
        'train.txt': 'train',
        'val.txt': 'test'
    }

    # Ensure target structure exists
    for split_name in split_mapping.values():
        os.makedirs(os.path.join(target_root, split_name, 'image'), exist_ok=True)
        os.makedirs(os.path.join(target_root, split_name, 'label'), exist_ok=True)

    # Process each split file
    for txt_file, target_subdir in split_mapping.items():
        txt_path = os.path.join(split_dir, txt_file)
        if not os.path.exists(txt_path):
            print(f"Warning: {txt_path} not found, skipping.")
            continue
            
        with open(txt_path, 'r') as f:
            stems = [line.strip() for line in f if line.strip()]
            
        print(f"Processing {txt_file} -> {target_subdir} ({len(stems)} files)")
        
        for stem in stems:
            # Source paths
            # Try jpg then png for image 
            src_img_jpg = os.path.join(source_root, 'image', f'{stem}.jpg')
            src_img_png = os.path.join(source_root, 'image', f'{stem}.png')
            
            # Label is png
            src_label = os.path.join(source_root, 'label', f'{stem}.png')
            
            # Determine correct image path
            if os.path.exists(src_img_jpg):
                src_img = src_img_jpg
                img_ext = '.jpg'
            elif os.path.exists(src_img_png):
                src_img = src_img_png
                img_ext = '.png'
            else:
                print(f"  Missing image for {stem}")
                continue
                
            if not os.path.exists(src_label):
                 print(f"  Missing label for {stem}")
                 continue

            # Destination paths
            dst_img = os.path.join(target_root, target_subdir, 'image', f'{stem}{img_ext}')
            dst_label = os.path.join(target_root, target_subdir, 'label', f'{stem}.png')
            
            # Copy
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_label, dst_label)

    print(f"Copy completed. Files are in {target_root}")

if __name__ == '__main__':
    main()
