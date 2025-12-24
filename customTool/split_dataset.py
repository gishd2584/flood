import os
import math

def main():
    root = 'data/guilinflood'
    image_dir = os.path.join(root, 'image')
    
    if not os.path.exists(image_dir):
        print(f"Error: {image_dir} not found.")
        return

    # Sort by filename
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    train_files = []
    val_files = []

    # 7:3 split with systematic sampling
    # We take indices i such that file goes to val if condition met
    # 3 out of 10 -> 30%
    # Pattern: T, T, V, T, T, V, T, T, V, T
    for i, f in enumerate(files):
        if i % 10 in [2, 5, 8]:
            val_files.append(f)
        else:
            train_files.append(f)

    # Output directory
    split_dir = os.path.join(root, 'splits')
    os.makedirs(split_dir, exist_ok=True)

    # Write stems (filenames without extension)
    def write_list(filename, file_list):
        with open(os.path.join(split_dir, filename), 'w') as f:
            # Writing only stems is standard for mmseg if suffixes are configured
            stems = [os.path.splitext(x)[0] for x in file_list]
            f.write('\n'.join(stems))
    
    write_list('train.txt', train_files)
    write_list('val.txt', val_files)

    print(f'Done. Total files: {len(files)}')
    print(f'Train files: {len(train_files)}')
    print(f'Test/Val files: {len(val_files)}')
    print(f'Saved to {split_dir}')

if __name__ == '__main__':
    main()
