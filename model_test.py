import glob
import cv2
from ultralytics import YOLO
import numpy as np
import torch
import os

DIR = "./dataset_yolo/images"
SAVE_DIR = "model_test"
os.path.isdir(SAVE_DIR) or os.makedirs(SAVE_DIR)

imgs = glob.glob(f'{DIR}/*.png')
model = YOLO("./best.pt")

for img_file in imgs:
    img = cv2.imread(img_file)
    result = model(img)[0]

    map = np.zeros(img.shape, dtype=np.uint8)
    
    if result.masks != None:
        for mask in result.masks:
            m = torch.squeeze(mask.data)
            composite = torch.stack((m, m, m), 2)
            tmp =  255 * composite.cpu().numpy().astype(np.uint8)
            resized = cv2.resize(tmp, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.bitwise_or(map, resized, map)
    cv2.addWeighted(img, 0.5, map, 0.5, 0, map)

    gray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    save_img = np.hstack(([map, img]))
    cv2.resize(save_img, (save_img.shape[1]//3, save_img.shape[0]//3), interpolation=cv2.INTER_LINEAR)

    #get filename from os.path
    filename = img_file.replace("\\", "/").split('/')[-1]
    save_path = f"{SAVE_DIR}/{filename}"
    cv2.imwrite(save_path, save_img)
    
