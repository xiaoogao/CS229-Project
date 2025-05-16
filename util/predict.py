from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import dlib
import os
import argparse
import shutil

if os.path.exists("multi_face_list.txt"):
    os.remove("multi_face_list.txt")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_dataset_with_ratio(dataset_list, ratio = 0.5, taret_file = '../dataset'):
    # 0: Male, 1: Female
    # Step 1: Reassemble the input images path
    for img_path, gender_idx in dataset_list:
        # add to dataset
        image_name = gender_idx + img_path.split("/")[-1]
        image_category = img_path.split("/")[-2]
        image_path = taret_file
        if gender_idx == 0:
            # Make sure the directory path exists.
            ensure_dir(os.path.join(image_path, image_category))
            shutil.copy(image_path, os.path.join(image_path, image_category, image_name))
        else:
            # add to dataset with a prob
            if np.random.rand() < ratio:
                # add to dataset
                shutil.copy()
        

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def detect_face(image_paths,  SAVE_DETECTED_AT, default_max_size=800,size = 300, padding = 0.25):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
    base = 2000  # largest width and height
    from tqdm import tqdm
    for index, image_path in tqdm(enumerate(image_paths)):
        category = image_path.split("/")[-2]
        if index % 1000 == 0:
            print('---%d/%d---' %(index, len(image_paths)))
        img = dlib.load_rgb_image(image_path)

        old_height, old_width, _ = img.shape

        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
        img = dlib.resize_image(img, rows=new_height, cols=new_width)

        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)
        # abnormal_face = []

        if num_faces != 1:
            print("Warning: %d faces detected in image %s" % (num_faces, image_path))
            with open("multi_face_list.txt", "a") as f_log:
                f_log.write(image_path + "\n")
            continue
        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            rect = detection.rect
            faces.append(sp(img, rect))
        images = dlib.get_face_chips(img, faces, size=size, padding = padding)
        for idx, image in enumerate(images):
            img_name = image_path.split("/")[-1]
            path_sp = img_name.split(".")
        
            ensure_dir(os.path.join(SAVE_DETECTED_AT, category))
            face_name = os.path.join(SAVE_DETECTED_AT,  category, img_name)
            dlib.save_image(image, face_name)
    
def predidct_age_gender_race(original_path="../../dataset/Raw_data", imgs_path = 'cropped_faces/', target_file = '../dataset'):
    for category in os.listdir(imgs_path):
        img_names = [os.path.join(imgs_path, category, x) for x in os.listdir(os.path.join(imgs_path, category))]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_fair_7 = torchvision.models.resnet34(pretrained=True)
        model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
        model_fair_7.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_7_20190809.pt'))
        model_fair_7 = model_fair_7.to(device)
        model_fair_7.eval()

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        ensure_dir(os.path.join(target_file, "Male", category))
        ensure_dir(os.path.join(target_file, "Female", category))
        for index, img_name in enumerate(img_names):
            if index % 1000 == 0:
                print("Predicting... {}/{}".format(index, len(img_names)))
            image = dlib.load_rgb_image(img_name)
            image = trans(image)
            image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
            image = image.to(device)

            # fair
            outputs = model_fair_7(image)
            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)

            # race_outputs = outputs[:7]
            gender_logits = outputs[7:9]
            gender_prob = torch.softmax(torch.tensor(gender_logits), dim=0).numpy()
            if gender_prob.max() < 0.6  and  gender_prob.max() > 0.4:
                continue

            gender_class = gender_prob.argmax()
            if gender_class == 0:
                shutil.move(os.path.join(original_path, category, img_name.split("/")[-1]), os.path.join(target_file, "Male", category, img_name.split("/")[-1]))
            else:   
                shutil.move(os.path.join(original_path, category, img_name.split("/")[-1]), os.path.join(target_file, "Female", category, img_name.split("/")[-1]))
        # result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
        # result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

if __name__ == "__main__":
    #Please create a csv with one column 'img_path', contains the full paths of all images to be analyzed.
    #Also please change working directory to this file.
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='../../dataset/Raw_dataset/', action='store',
                        help='path of the input directory')
    parser.add_argument('--target_file', default='../../dataset', type=str,
                        help='target file to save the images')
    dlib.DLIB_USE_CUDA = True

    print("using CUDA?: %s" % dlib.DLIB_USE_CUDA)
    args = parser.parse_args()
    if not os.path.exists(args.target_file):
        os.mkdir(args.target_file)

    SAVE_DETECTED_AT = os.path.join(args.target_file, 'cropped_faces')
    ensure_dir(SAVE_DETECTED_AT)
    imgs_path = args.file_path
    imgs = []
    for occupation in os.listdir(imgs_path):
        for img in os.listdir(os.path.join(imgs_path, occupation)):
            if img.endswith('.jpg') or img.endswith('.png'):
                imgs.append(os.path.join(imgs_path, occupation, img))
    # breakpoint()
    detect_face(imgs, SAVE_DETECTED_AT)
    # print("detected faces are saved at ", SAVE_DETECTED_AT)
    #Please change test_outputs.csv to actual name of output csv. 
    predidct_age_gender_race(args.file_path, SAVE_DETECTED_AT, target_file=args.target_file)
