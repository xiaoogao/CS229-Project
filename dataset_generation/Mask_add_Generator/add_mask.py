import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

def mask_face_auto(pil_img):
    # Detect face and mask automatically using OpenCV Haar cascade
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return None  # No face detected, signal for manual intervention
    for (x, y, w, h) in faces:
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)
    masked_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    return masked_pil

def mask_folder_with_manual(input_root, output_root):
    for dirpath, dirnames, filenames in os.walk(input_root):
        for fname in tqdm(filenames, desc=f"Masking {dirpath}"):
            if not fname.lower().endswith(('.png','.jpg','.jpeg')):
                continue
            in_path = os.path.join(dirpath, fname)
            rel_dir = os.path.relpath(dirpath, input_root)
            out_dir = os.path.join(output_root, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)
            try:
                img = Image.open(in_path).convert("RGB")
                masked_img = mask_face_auto(img)
                if masked_img is not None:
                    masked_img.save(out_path)
                else:
                    # Face not found, launch manual GUI to cutout
                    print(f"No face found for {in_path}, please cutout manually.")
                    manual_cutout_and_save(img, out_path)
            except Exception as e:
                print(f"Failed on {in_path}: {e}")

