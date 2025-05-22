import argparse
import torch
import torch.nn as nn
import numpy as np
import timm
import json
import torchvision
from torchvision import datasets, transforms
from backbones import resnet, swin_transformer
import os
import sys
from util.helper import FocalLoss
from tqdm import tqdm

def preprocess_resnet50(num_classes=10, pretrain_path='./checkpoint/resnet50-19c8e357.pth'):
    """Preprocess the input for ResNet-50 and freeze all layers except the final fc."""
    model = resnet.resnet50(num_classes=1000).cuda()
    model.load_state_dict(torch.load(pretrain_path, weights_only=True))

    # Freeze all layers except the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes).cuda()
    return model

def preprocess_swin_transformer(num_classes=10):
    """Preprocess the input for Swin Transformer and freeze all layers except the final head."""
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=1000).cuda()

    # Freeze all layers except the final head
    for param in model.parameters():
        param.requires_grad = False
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes).cuda()
    return model

def main():
    parser = argparse.ArgumentParser(description="Train classification model on custom dataset.")
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'swin'], help='Model to use (default: resnet50)')
    parser.add_argument('--data_root', type=str, default='./dataset_100', help='Root directory of train/val split')
    parser.add_argument('--save_path', type=str, default='./checkpoint/ResNet50_10occupation.pth', help='Path to save best model checkpoint')
    parser.add_argument('--pretrain_path', type=str, default='./checkpoint/resnet50-19c8e357.pth', help='Pretrained weights path for ResNet50')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training/validation')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--use_focal_loss', action='store_true', default=False, help='Use focal loss in addition to cross entropy')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for this project')

    # Data transforms
    data_transform = {
        "train": transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
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

    # Dataset loading
    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'val')
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transform['val'])
    category_to_label = train_dataset.class_to_idx
    label_to_category = dict((val, key) for key, val in category_to_label.items())
    # Save label mapping for evaluation
    with open('class_label.json', 'w') as json_file:
        json.dump(label_to_category, json_file, indent=4)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    val_num = len(val_dataset)
    batch_num = len(train_loader)

    # Model selection
    num_classes = len(category_to_label)
    if args.model == 'resnet50':
        model = preprocess_resnet50(num_classes=num_classes, pretrain_path=args.pretrain_path)
    elif args.model == 'swin':
        model = preprocess_swin_transformer(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model {args.model}")
    model.to(device)

    # Loss and optimizer
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    if args.use_focal_loss:
        focal_loss = FocalLoss(gamma=2, alpha=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    best_acc = 0.0
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
        print(f'Epoch[{epoch + 1}/{args.epochs}] train_loss: {total_loss:.3f}, train_accuracy: {train_accuracy:.3f}')

        # Validation
        model.eval()
        val_bar = tqdm(val_loader, file=sys.stdout)
        acc = 0
        val_total = 0
        with torch.no_grad():
            for data in val_bar:
                val_img, val_label = data
                predict = model(val_img.to(device))
                output = torch.max(predict, dim=1)[1]
                acc += torch.eq(output, val_label.to(device)).sum().item()
                val_total += val_label.size(0)
        val_accuracy = acc / val_total
        val_bar.set_description(f"valid epoch[{epoch + 1}/{args.epochs}] acc: {val_accuracy:.3f}")

        print(f'[epoch {epoch + 1}] train_loss: {total_loss:.3f}  train_accuracy: {train_accuracy:.3f}  val_accuracy: {val_accuracy:.3f}')

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), args.save_path)
            print(f"Best model updated: epoch {epoch+1}, val_acc={val_accuracy:.3f}")

    print('Finish Training!')

if __name__ == "__main__":
    main()