import os
import shutil
import random
import pandas as pd
import argparse
from PIL import Image

def check_images(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(dirpath, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()  # Don't decode full image, just check header
                except Exception as e:
                    print(f"Corrupted: {fpath} ({e})")
                    os.remove(fpath)  # Or move to a backup folder


def ensure_dir(path):
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

def get_sorted_files(folder):
    # Return a sorted list of image files in the folder
    return sorted([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

def split_and_sample_gender(
    folder_A, folder_B, output_root, 
    train_ratio=0.8, female_train_ratio=1.0, 
    seed=42
):
    """
    Merge datasets from two sources, split into train/val, and sample female images in train.
    Args:
        folder_A, folder_B: Source dataset directories.
        output_root: Output dataset root directory.
        train_ratio: Proportion of data used for training (e.g., 0.8 for 80% train, 20% val).
        female_train_ratio: Proportion of female images to keep in train split (e.g., 1.0=all, 0.5=half, 0.0=none).
        seed: Random seed for reproducibility.
    """
    # Check and clean images in both folders
    check_images(folder_A)
    check_images(folder_B)
    
    random.seed(seed)
    meta = []

    # Get all subfolders in both sources
    subfolders_A = set(os.listdir(folder_A))
    subfolders_B = set(os.listdir(folder_B))
    common_subfolders = subfolders_A & subfolders_B

    for subfolder in common_subfolders:
        path_A = os.path.join(folder_A, subfolder)
        path_B = os.path.join(folder_B, subfolder)
        all_files_A = get_sorted_files(path_A)
        all_files_B = get_sorted_files(path_B)

        # Collect (path, gender, original filename) for all images
        all_files = []
        for img_file in all_files_A:
            abs_path = os.path.join(path_A, img_file)
            gender = "male" if img_file.startswith("0_") else "female" if img_file.startswith("1_") else "unknown"
            all_files.append((abs_path, gender, img_file))
        for img_file in all_files_B:
            abs_path = os.path.join(path_B, img_file)
            gender = "male" if img_file.startswith("0_") else "female" if img_file.startswith("1_") else "unknown"
            all_files.append((abs_path, gender, img_file))
        
        # Separate into male and female groups
        males = [item for item in all_files if item[1] == "male"]
        females = [item for item in all_files if item[1] == "female"]
        if len(males) == 0 or len(females) == 0:
            print(f"Warning: {subfolder} has only one gender's images, skipped.")
            continue

        # Shuffle each gender group
        random.shuffle(males)
        random.shuffle(females)

        # Split each gender group into train/val
        n_train_male = int(len(males) * train_ratio)
        n_train_female = int(len(females) * train_ratio)
        train_males = males[:n_train_male]
        val_males = males[n_train_male:]
        train_females = females[:n_train_female]
        val_females = females[n_train_female:]

        # Sample a proportion of female images for the train split
        n_sample_female = int(len(train_females) * female_train_ratio)
        sampled_train_females = (
            random.sample(train_females, n_sample_female)
            if n_sample_female < len(train_females)
            else train_females
        )

        # Combine train and val splits, shuffle to remove order bias
        train_files = train_males + sampled_train_females
        val_files = val_males + val_females
        random.shuffle(train_files)
        random.shuffle(val_files)

        # Save images to the appropriate directories and record metadata
        for split, file_list in zip(['train', 'val'], [train_files, val_files]):
            out_dir = os.path.join(output_root, split, subfolder)
            ensure_dir(out_dir)
            for idx, (src_path, gender, orig_fname) in enumerate(file_list, 1):
                dst_fname = f"{('0' if gender == 'male' else '1')}_{idx:08d}.png"
                dst_path = os.path.join(out_dir, dst_fname)
                shutil.copyfile(src_path, dst_path)
                meta.append({
                    "profession": subfolder,
                    "split": split,
                    "gender": gender,
                    "src_file": src_path,
                    "dst_file": dst_path,
                    "orig_fname": orig_fname
                })
        print(f"Processed '{subfolder}' | train: {len(train_files)}, val: {len(val_files)}, female_in_train: {len(sampled_train_females)}")

    # Save metadata for further analysis
    pd.DataFrame(meta).to_csv(os.path.join(output_root, "dataset_metadata.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="Merge datasets, split train/val, and sample female images in train split.")
    parser.add_argument("--folder_A", type=str, required=True, help="Path to first source dataset (e.g. Crawler_Dataset)")
    parser.add_argument("--folder_B", type=str, required=True, help="Path to second source dataset (e.g. SDxl_data)")
    parser.add_argument("--output_root", type=str, required=True, help="Output directory for the new merged dataset")
    parser.add_argument("--train_ratio", type=float, default=0.85, help="Ratio of data used for training (default: 0.8)")
    parser.add_argument("--female_train_ratio", type=float, default=1.0, help="Proportion of female images to keep in train split (default: 1.0)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    args = parser.parse_args()

    split_and_sample_gender(
        args.folder_A,
        args.folder_B,
        args.output_root,
        train_ratio=args.train_ratio,
        female_train_ratio=args.female_train_ratio,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()