import argparse
import torch
import torch.nn as nn
import timm
import json
from torchvision import datasets, transforms
import os
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path for import and resource access
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
torch.autograd.set_detect_anomaly(True)

from backbones import resnet, swin_transformer
from util.helper import FocalLoss

def preprocess_resnet10(num_classes=10, pretrain_path=None):
    model = resnet.resnet10(num_classes=1000).cuda()
    if pretrain_path:
        model.load_state_dict(torch.load(pretrain_path, weights_only=True))
    model.fc = nn.Linear(model.fc.in_features, num_classes).cuda()
    return model

def save_loss_acc_curves(train_losses, val_losses, train_accs, val_accs, save_prefix):
    """
    Save training/validation loss and accuracy curves using seaborn for better aesthetics.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        train_accs (list): List of training accuracies per epoch.
        val_accs (list): List of validation accuracies per epoch.
        save_prefix (str): The file path prefix to save figures (no extension).
    """
    sns.set_theme(style="whitegrid", font_scale=1.2)
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss Curve
    plt.figure(figsize=(7, 5))
    sns.lineplot(x=epochs, y=train_losses, label='Train Loss', marker="o", markersize=3, linewidth=2)
    sns.lineplot(x=epochs, y=val_losses, label='Val Loss', marker="s", markersize=3, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png", dpi=200)
    plt.close()

    # Plot Accuracy Curve
    plt.figure(figsize=(7, 5))
    sns.lineplot(x=epochs, y=train_accs, label='Train Accuracy', marker="o", markersize=3, linewidth=2)
    sns.lineplot(x=epochs, y=val_accs, label='Val Accuracy', marker="s", markersize=3, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(frameon=True, loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_acc.png", dpi=200)
    plt.close()


def get_experiment_dir(base_path, model, data_root):
    dataset_name = os.path.basename(os.path.normpath(data_root))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_path, f"{model}_{dataset_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def main():
    parser = argparse.ArgumentParser(description="Train classification model on custom dataset.")
    parser.add_argument('--model', type=str, default='resnet10', help='Model to use (default: resnet18)')
    parser.add_argument('--data_root', type=str, default='../dataset/dataset_100', help='Root directory of train/val split')
    parser.add_argument('--save_path', type=str, default='../checkpoint', help='Root path to save experiment folders')
    parser.add_argument('--pretrain_path', type=str, default=None, help='Pretrained weights path for ResNet18')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training/validation')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--use_focal_loss', action='store_true', default=False, help='Use focal loss in addition to cross entropy')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    args = parser.parse_args()

    # ==== Exp Catalog ====
    exp_dir = get_experiment_dir(args.save_path, args.model, args.data_root)
    print(f'All experiment results will be saved in: {exp_dir}')

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # ==== Data transforms ====
    data_transform = {
        "train": transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # ==== Dataset ====
    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'val')
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform['val'])
    category_to_label = train_dataset.class_to_idx
    label_to_category = dict((val, key) for key, val in category_to_label.items())
    with open(os.path.join(exp_dir, 'class_label.json'), 'w') as json_file:
        json.dump(label_to_category, json_file, indent=4)

    # ==== DataLoader ====
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # ==== Model ====
    num_classes = len(category_to_label)
    if args.model == 'resnet10':
        model = preprocess_resnet10(num_classes=num_classes, pretrain_path=args.pretrain_path)
    else:
        raise ValueError(f"Unknown model {args.model}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ==== Loss and optimizer ====
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.15)
    if args.use_focal_loss:
        focal_loss = FocalLoss(gamma=2, alpha=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # ==== Training loop ====
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.epochs):
        model.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        total_loss = 0.0
        correct = 0
        total = 0
        for step, data in enumerate(train_bar):
            optimizer.zero_grad()
            img, label = data
            predict = model(img.to(device))
            if not args.use_focal_loss:
                loss = loss_function(predict, label.to(device))
            else:
                loss = focal_loss(predict, label.to(device)) + loss_function(predict, label.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Train accuracy
            output = torch.max(predict, dim=1)[1]
            correct += torch.eq(output, label.to(device)).sum().item()
            total += label.size(0)
            train_bar.set_description(f"train epoch[{epoch + 1}/{args.epochs}] acc: {correct/total:.3f}")

        train_accuracy = correct / total
        total_loss = total_loss / total
        train_losses.append(total_loss)
        train_accs.append(train_accuracy)

        # ==== Validation ====
        model.eval()
        val_bar = tqdm(val_loader, file=sys.stdout)
        acc = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for data in val_bar:
                val_img, val_label = data
                predict = model(val_img.to(device))
                if not args.use_focal_loss:
                    loss = loss_function(predict, val_label.to(device))
                else:
                    loss = focal_loss(predict, val_label.to(device)) + loss_function(predict, val_label.to(device))
                val_loss += loss.item()
                output = torch.max(predict, dim=1)[1]
                acc += torch.eq(output, val_label.to(device)).sum().item()
                val_total += val_label.size(0)
        val_accuracy = acc / val_total
        val_loss = val_loss / val_total
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        print(f'Epoch[{epoch + 1}/{args.epochs}] train_loss: {total_loss:.3f}, train_accuracy: {train_accuracy:.3f}, val_loss: {val_loss:.3f}  val_accuracy: {val_accuracy:.3f}')

        # ==== logs ====
        with open(os.path.join(exp_dir, "log.txt"), "a") as f:
            f.write(f"Epoch[{epoch + 1}/{args.epochs}] train_loss: {total_loss:.3f}, train_acc: {train_accuracy:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_accuracy:.3f}\n")

        # ==== save best model ====
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            save_name = os.path.join(exp_dir,"best_model.pth")
            torch.save(model.state_dict(), save_name)
            print(f"Best model updated: epoch {epoch+1}, val_acc={val_accuracy:.3f}")

    print('Finish Training!')
    # ==== save loss/acc ====
    save_loss_acc_curves(train_losses, val_losses, train_accs, val_accs, os.path.join(exp_dir, args.model))


if __name__ == "__main__":
    main()
