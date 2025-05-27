import os
import sys
import argparse
import torch

# Add project root to sys.path for import and resource access
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
torch.autograd.set_detect_anomaly(True)

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import face_recognition
import timm
from torchvision import transforms
from backbones import resnet
from util.cam_utils import GradCAM, GradCAMPlusPlus

def parse_args():
    parser = argparse.ArgumentParser(
        description="Face Attribution Analysis with GradCAM/GradCAM++"
    )
    parser.add_argument('--data_root', type=str, required=True,
                        help="Path to the input dataset (organized as data_root/profession/image.jpg)")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument('--model_name', type=str, choices=['resnet10', 'resnet18', 'resnet50', 'swin'],
                        required=True, help="Model architecture")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="Number of classes (default: 10)")
    parser.add_argument('--use_gradcampp', action='store_true',
                        help="Use GradCAM++ if set; otherwise, use GradCAM")
    parser.add_argument('--threshold', type=float, default=0.2,
                        help="Threshold for activation mask (default: 0.2)")
    parser.add_argument('--output_csv', type=str, default="face_attribution_ratio.csv",
                        help="Path to save the results csv file")
    parser.add_argument('--output_plot', type=str, default="face_attribution_ratio_barplot_seaborn.png",
                        help="Path to save the output barplot image")
    return parser.parse_args()

def preprocess_resnet10(num_classes=10):
    model = resnet.resnet10(num_classes=num_classes).to(device)
    return model

def preprocess_resnet18(num_classes=10):
    model = resnet.resnet18(num_classes=num_classes).to(device)
    return model

def preprocess_resnet50(num_classes=10):
    model = resnet.resnet50(num_classes=num_classes).to(device)
    return model

def preprocess_swin_transformer(num_classes=10):
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes).to(device)
    return model

def swin_reshape_transform(x):
    B, L, C = x.shape
    H = W = int(L ** 0.5)
    return x.permute(0, 2, 1).contiguous().view(B, C, H, W)

def get_gender_label(path):
    # Assumes gender info encoded in filename (customize for your setup)
    if os.path.basename(path).startswith("0_"):
        return "male"
    else:
        return "female"

def get_profession_label(path):
    # Assumes second-level directory is the profession
    return os.path.basename(os.path.dirname(path))

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model Preparation ----
    if args.model_name == "resnet50":
        model = preprocess_resnet50(num_classes=args.num_classes)
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        target_layers = [model.layer4[-1]]
        is_transformer = False
    elif args.model_name == "resnet18":
        model = preprocess_resnet18(num_classes=args.num_classes)
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        target_layers = [model.layer4[-1]]
        is_transformer = False
    elif args.model_name == "resnet10":
        model = preprocess_resnet10(num_classes=args.num_classes)
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        target_layers = [model.layer4[-1]]
        is_transformer = False
    elif args.model_name == "swin":
        model = preprocess_swin_transformer(num_classes=args.num_classes)
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        target_layers = [model.layers[-1].blocks[-1].norm2]
        is_transformer = True
    else:
        raise ValueError("Unsupported model type.")

    model.eval()

    # ---- Image Transform ----
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ---- GradCAM/GradCAM++ ----
    if args.use_gradcampp:
        CAM = GradCAMPlusPlus(
            model, target_layers,
            reshape_transform=swin_reshape_transform if is_transformer else None,
            use_cuda=(device.type == "cuda")
        )
    else:
        CAM = GradCAM(
            model, target_layers,
            reshape_transform=swin_reshape_transform if is_transformer else None,
            use_cuda=(device.type == "cuda")
        )

    # ---- Main Loop ----
    records = []
    for profession in os.listdir(args.data_root):
        prof_dir = os.path.join(args.data_root, profession)
        if not os.path.isdir(prof_dir):
            continue
        for fname in tqdm(os.listdir(prof_dir), desc=f"{profession}"):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(prof_dir, fname)
            # 1. Face Detection
            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil)
            face_locations = face_recognition.face_locations(img_np)
            if len(face_locations) == 0:
                continue  # No face detected, skip

            # 2. GradCAM
            input_tensor = transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            mask = CAM(input_tensor, target_category=pred_class)[0]  # shape: (H, W)
            # mask is already normalized to 0-1

            # 3. Compute attribution ratio inside face region
            act_mask = mask > args.threshold
            total_act = act_mask.sum()
            face_act = 0
            for (top, right, bottom, left) in face_locations:
                face_box = act_mask[top:bottom, left:right]
                face_act += face_box.sum()
            face_ratio = face_act / total_act if total_act > 0 else 0

            # 4. Record results
            gender = get_gender_label(img_path)
            records.append({
                "img_path": img_path,
                "profession": profession,
                "gender": gender,
                "face_ratio": face_ratio,
                "pred_class": pred_class
            })

    # ---- Save results ----
    df = pd.DataFrame(records)
    df['face_ratio_pct'] = df['face_ratio'] * 100
    df.to_csv(args.output_csv, index=False)
    print(f"Saved attribution stats to {args.output_csv}")

    # ---- Bar Plot ----
    plt.rcParams['font.family'] = 'Arial'

    pivot = df.groupby(['profession', 'gender'])['face_ratio'].mean().unstack()
    pivot.plot(kind='bar', figsize=(12,6), ylabel="Mean Face Attribution Ratio")
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid", font_scale=1.2)
    ax = sns.barplot(
        data=df,
        x="profession",
        y="face_ratio_pct",
        hue="gender",     
        capsize=0.15,
        errorbar=None,
        palette="Set2"
    )
    ax.set_xlabel("Occupation", fontsize=16)
    ax.set_ylabel("Mean Face Attribution Ratio (%)", fontsize=16)
    ax.set_yticks(np.arange(0, 31, 5)) 
    plt.legend(title="Gender", fontsize=14, title_fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300)
    plt.show()
