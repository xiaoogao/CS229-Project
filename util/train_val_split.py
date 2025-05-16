import numpy as np
import os
import random

import shutil

def main(dataset_path, save_path, train_split_ratio =0.8,):
    # Set the random seed for reproducibility
    random.seed(42)

    # Get the list of all image files in the dataset directory
    all_images = []
    for root, _, files in os.walk(dataset_path):
        for occupation in os.listdir(root):
            occupation_path = os.path.join(root, occupation)
            if not os.path.isdir(occupation_path):
                continue
            for file in os.listdir(occupation_path):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(occupation_path, file))

    # Shuffle the list of images
    random.shuffle(all_images)

    # Calculate the number of training images
    num_train = int(len(all_images) * train_split_ratio)

    # Split the images into training and validation sets
    train_images = all_images[:num_train]
    val_images = all_images[num_train:]

    # Move the images to their respective directories
    for img in train_images:
        category = os.path.basename(os.path.dirname(img))
        os.makedirs(os.path.join(save_path, 'train', category), exist_ok=True)
        shutil.copy(img, os.path.join(save_path, 'train', category, os.path.basename(img)))

    for img in val_images:
        category = os.path.basename(os.path.dirname(img))
        os.makedirs(os.path.join(save_path, 'val', category), exist_ok=True)
        shutil.copy(img, os.path.join(save_path, 'val', category, os.path.basename(img)))

if __name__ == "__main__":
    dataset_path = '../dataset/Raw_dataset/'
    save_path = '../dataset/Train_val_split/'
    main(dataset_path, save_path)