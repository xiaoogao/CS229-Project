import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from facenet_pytorch import MTCNN
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

def mask_face_auto(pil_img):
    boxes, _ = mtcnn.detect(pil_img)
    if boxes is None or len(boxes) == 0:
        return None
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def mask_folder_only_masked(input_root, output_root):
    """
    Only copy images where a face was detected and masked (train).
    For val, just copy.
    Directory structure is preserved.
    """
    success, total, val_count, train_count = 0, 0, 0, 0
    for dirpath, dirnames, filenames in os.walk(input_root):
        fnames = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for fname in tqdm(fnames, desc=f"Processing {dirpath}", leave=False):
            in_path = os.path.join(dirpath, fname)
            rel_dir = os.path.relpath(dirpath, input_root)
            out_dir = os.path.join(output_root, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)
            total += 1
            dirpath_lower = dirpath.lower()
            try:
                if "train" in dirpath_lower:
                    img = Image.open(in_path).convert("RGB")
                    masked_img = mask_face_auto(img)
                    train_count += 1
                    if masked_img is not None:
                        masked_img.save(out_path)
                        success += 1
                elif "val" in dirpath_lower:
                    shutil.copy2(in_path, out_path)
                    val_count += 1
            except (OSError, Exception) as e:
                print(f"[Error] Failed on {in_path}: {e}")
    print(f"\nSummary:")
    print(f"  Train images processed: {train_count}, Masked (face detected): {success}")
    print(f"  Val images copied:      {val_count}")
    print(f"  Total images seen:      {total}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch mask faces in images under a directory, only saving masked images.")
    parser.add_argument("--input_root", type=str, required=True, help="Input directory of images")
    parser.add_argument("--output_root", type=str, default=None, help="Optional: Output directory")
    args = parser.parse_args()
    # Output path
    if args.output_root is None:
        parent, base = os.path.split(os.path.abspath(args.input_root.rstrip("/\\")))
        output_root = os.path.join(parent, base + "_mask")
    else:
        output_root = args.output_root

    if os.path.exists(output_root):
        print(f"[Warning] Output directory {output_root} already exists. Remove or choose another path.")
        return

    os.makedirs(output_root, exist_ok=True)
    print(f"Input:  {args.input_root}\nOutput: {output_root}\n")
    mask_folder_only_masked(args.input_root, output_root)
    print("Done.")

if __name__ == "__main__":
    main()


# python add_mask_dlib.py --input_root ../../dataset/dataset_0/
# python add_mask_dlib.py --input_root ../../dataset/dataset_100/
