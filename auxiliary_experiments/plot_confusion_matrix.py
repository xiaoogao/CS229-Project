import os
import sys
# Add project root to sys.path for import and resource access
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.helper import ConfusionMatrix
import argparse
import torch
import numpy as np
import timm
import json
from torchvision import transforms
from PIL import Image
from backbones import resnet
from tqdm import tqdm

def preprocess_resnet10(num_classes=10):
    model = resnet.resnet10(num_classes=num_classes).cuda()
    return model

def preprocess_resnet50(num_classes=10):
    model = resnet.resnet50(num_classes=num_classes).cuda()
    return model

def preprocess_swin_transformer(num_classes=10):
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes).cuda()
    return model

def load_class_label(label_path):
    with open(label_path, "r") as f:
        label_to_category = json.load(f)
    # Ensure index sorted order
    labels = [label_to_category[str(i)] if str(i) in label_to_category else str(i) for i in range(len(label_to_category))]
    return labels

def get_image_paths_and_labels(root_dir, label_to_idx):
    img_paths, labels = [], []
    for cls_name, cls_idx in label_to_idx.items():
        cls_dir = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)
    return img_paths, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'swin', 'resnet10'])
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--label_json', type=str, default='../class_label.json')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--save_fig', type=str, default='confmat.png')
    args = parser.parse_args()

    # Load labels
    labels = load_class_label(args.label_json)
    num_classes = len(labels)
    label_to_idx = {v: k for k, v in enumerate(labels)}  # category name -> int idx

    weight_dir = os.path.dirname(args.weights)
    save_path = os.path.join(weight_dir, args.save_fig)
    log_path = os.path.join(weight_dir, "confmat_log.txt")

    # Model
    if args.model == "resnet50":
        model = preprocess_resnet50(num_classes=num_classes)
    elif args.model == "swin":
        model = preprocess_swin_transformer(num_classes=num_classes)
    elif args.model == "resnet10":
        model = preprocess_resnet10(num_classes=num_classes)
    else:
        raise ValueError("Unsupported model")

    model.load_state_dict(torch.load(args.weights, map_location="cuda", weights_only=True))
    model.eval()

    # Preprocessing
    data_transform = transforms.Compose([
        transforms.Resize(args.img_size + 32),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Collect data
    img_paths, gt_labels = get_image_paths_and_labels(args.data_dir, {name: i for i, name in enumerate(labels)})

    # Run prediction and update confusion matrix
    cm = ConfusionMatrix(num_classes=num_classes, labels=labels)
    batch_size = 32
    all_preds, all_gts = [], []

    for idx in tqdm(range(0, len(img_paths), batch_size), desc="Predicting"):
        batch_paths = img_paths[idx:idx+batch_size]
        imgs = []
        gts = gt_labels[idx:idx+batch_size]
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(data_transform(img))
        imgs = torch.stack(imgs).cuda()
        with torch.no_grad():
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_gts.extend(gts)

    cm.update(all_preds, all_gts)
    cm.summary()
    cm.plot(save_path=save_path)
    print(f"Confusion matrix saved to {save_path}")

    # Save confusion matrix log
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    cm.summary()
    sys.stdout = old_stdout
    with open(log_path, "w") as f:
        f.write(mystdout.getvalue())
    print(f"Confusion matrix log saved to {log_path}")

if __name__ == "__main__":
    main()

# python plot_confusion_matrix.py --model resnet50 --data_dir ./dataset/dataset_100/val/ --weights ./checkpoint/resnet50