import numpy as np
import cv2
import torch


def get_angle(rect):
    if(rect[1][0] <rect[1][1]):
        angle = 90 -rect[2]
    else:
        angle = -rect[2]
    #각도 범위 -133~ +43
    #43도 초과 45도 이하일 때(집을 수 없음)
    #43도 고정 2도 오차 믿어보자
    if angle > 43 and angle <=45:
        angle = 43
    elif angle > 45:
        angle = angle - 180
    return angle

def get_location(img, min_area = 310000):
    global model
    result = model.predict(source = img, verbose=False)[0]
    map = np.zeros(img.shape, dtype=np.uint8)
    if result.masks != None:
        for mask in result.masks:
            m = torch.squeeze(mask.data)
            composite = torch.stack((m, m, m), 2)
            tmp =  255 * composite.cpu().numpy().astype(np.uint8)
            resized = cv2.resize(tmp, (img.shape[1], img.shape[0]), interpolation=cv2.INTERSECT_NONE)
            cv2.bitwise_or(map, resized, map)

    gray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)

    contours, _ =cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda x:  cv2.contourArea(x) > 30000, contours))

    locations = []
    if len(contours) == 0:
        return None
    
    areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        areas.append(cv2.contourArea(contour))

        location = {
            "center": np.int0(rect[0]),
            "angle" :get_angle(rect),
            "box" : np.int0(box)
        }
        locations.append(location)

    if len(areas) ==0:
        return None
    
    max_area_index = np.argmax(areas)

    return locations[max_area_index]


import os
from ultralytics import YOLO
import glob
DIR = "./dataset_yolo/images"
SAVE_DIR = "location_test"
os.path.isdir(SAVE_DIR) or os.makedirs(SAVE_DIR)

imgs = glob.glob(f'{DIR}/*.png')
model = YOLO("./best.pt")

for img_file in imgs:
    img = cv2.imread(img_file)
    location = get_location(img)
    
    if location is not None:
        cv2.drawContours(img, [location['box']], 0, (0,255, 0), 2)
        cv2.circle(img, location['center'], 2, (0, 255, 0), -1)

    filename = img_file.replace("\\", "/").split('/')[-1]
    save_path = f"{SAVE_DIR}/{filename}"
    cv2.imwrite(save_path, img)