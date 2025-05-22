import argparse
import torch
import numpy as np
import cv2
import timm
import json
from torchvision import transforms
from PIL import Image
from util.cam_utils import GradCAM, GradCAMPlusPlus
from backbones import resnet

def preprocess_resnet50(num_classes=10):
    model = resnet.resnet50(num_classes=num_classes).cuda()
    return model

def preprocess_swin_transformer(num_classes=10):
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes).cuda()
    return model

def swin_reshape_transform(x):
    B, L, C = x.shape
    H = W = int(L ** 0.5)
    return x.permute(0, 2, 1).contiguous().view(B, C, H, W)

def evaluate(
    model, 
    image_tensor, 
    device,
    class_names=None,
    use_gradcam=False,
    use_gradcampp=False,
    gradcam_target_layers=None,
    save_path=None,
    target_category=None,
    show_origin=True,
    is_transform_based_model=False
):
    if not (use_gradcam or use_gradcampp):
        raise ValueError("You must set use_gradcam or use_gradcampp to True!")
    assert gradcam_target_layers is not None, "Target layers must be specified for Grad-CAM."
    
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred_class = torch.argmax(output, dim=1).item()
    
    cam_category = pred_class if target_category is None else target_category
    class_name = class_names[cam_category] if class_names else f"class_{cam_category}"
    
    img_np = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    
    def apply_cam(mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        return np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)
    
    vis_list = []
    titles = []
    
    if show_origin:
        vis_list.append(img_np)
        titles.append("Original")

    if use_gradcam:
        cam = GradCAM(model=model,
                      target_layers=gradcam_target_layers,
                      use_cuda=device.type == 'cuda',
                      reshape_transform=swin_reshape_transform if is_transform_based_model else None)
        mask = cam(image_tensor, target_category=cam_category)[0]
        vis_list.append(apply_cam(mask))
        titles.append("Grad-CAM")

    if use_gradcampp:
        campp = GradCAMPlusPlus(model=model, 
                                target_layers=gradcam_target_layers, 
                                use_cuda=device.type == 'cuda',
                                reshape_transform=swin_reshape_transform if is_transform_based_model else None)
        maskpp = campp(image_tensor, target_category=cam_category)[0]
        vis_list.append(apply_cam(maskpp))
        titles.append("Grad-CAM++")

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(vis_list), figsize=(6 * len(vis_list), 5))
    if len(vis_list) == 1:
        axs = [axs]
    for ax, img, title in zip(axs, vis_list, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Saved] Visualization to {save_path}")
    plt.show()
    return pred_class, class_name

def main():
    parser = argparse.ArgumentParser(description="Visualize GradCAM/GradCAM++ for ResNet/Swin models.")
    parser.add_argument('--model', type=str, choices=['resnet50', 'swin_tiny'], default='resnet50', help='Model type')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--img', type=str, required=True, help='Image path')
    parser.add_argument('--labels', type=str, required=True, help='class_label.json path')
    parser.add_argument('--save', type=str, default='CAM_Visualization.png', help='Save path for output')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--use_gradcam', action='store_true', help='Use GradCAM')
    parser.add_argument('--use_gradcampp', action='store_true', help='Use GradCAM++')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'resnet50':
        model = preprocess_resnet50(num_classes=args.num_classes)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        target_layers = [model.layer4[-1]]
        is_transform_based_model = False
    elif args.model == 'swin_tiny':
        model = preprocess_swin_transformer(num_classes=args.num_classes)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        target_layers = [model.layers[-1].blocks[-1].norm2]
        is_transform_based_model = True
    else:
        raise ValueError("Unsupported model type.")

    img = Image.open(args.img).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)

    with open(args.labels, "r") as f:
        label_to_category = json.load(f)

    pred_class, class_name = evaluate(
        model=model,
        image_tensor=img_tensor,
        device=device,
        class_names=list(label_to_category.values()),
        use_gradcam=args.use_gradcam,
        use_gradcampp=args.use_gradcampp,
        gradcam_target_layers=target_layers,
        save_path=args.save,
        is_transform_based_model=is_transform_based_model
    )

    print(f"Predicted Class: {pred_class} ({class_name})")

if __name__ == "__main__":
    main()
