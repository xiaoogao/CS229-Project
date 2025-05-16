import os
import shutil

def ensure_dir(path):
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

def get_sorted_files(folder):
    # Return a sorted list of image files in the folder
    return sorted([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

def merge_images(folder_A, folder_B, output_root):
    # Get all subfolders in both folder A and folder B
    subfolders_A = set(os.listdir(folder_A))
    subfolders_B = set(os.listdir(folder_B))

    # Only process subfolders that exist in both A and B
    common_subfolders = subfolders_A & subfolders_B

    for subfolder in common_subfolders:
        path_A = os.path.join(folder_A, subfolder)
        path_B = os.path.join(folder_B, subfolder)
        output_dir = os.path.join(output_root, subfolder)
        ensure_dir(output_dir)

        count = 1  # Start numbering from 1

        # Copy and rename images from folder A
        for img_file in get_sorted_files(path_A):
            src = os.path.join(path_A, img_file)
            dst = os.path.join(output_dir, f"{count:08d}.png")
            shutil.copyfile(src, dst)
            count += 1

        # Copy and rename images from folder B, continuing the numbering
        for img_file in get_sorted_files(path_B):
            src = os.path.join(path_B, img_file)
            dst = os.path.join(output_dir, f"{count:08d}.png")
            shutil.copyfile(src, dst)
            count += 1

        print(f"Merged '{subfolder}' into '{output_dir}' with {count - 1} files.")

if __name__ == "__main__":
    folder_A = "./Crawler_Dataset"  
    folder_B = "./SDxl_data"
    output_root = "../dataset/Raw_dataset"  # Output root directory

    merge_images(folder_A, folder_B, output_root)

