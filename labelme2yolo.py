import glob
import os
import json
import shutil
import cv2

SRC_DIR = "dataset"
DST_DIR = "dataset_yolo"
LABELS = ["OK"]

os.path.isdir(DST_DIR) or os.makedirs(DST_DIR)


jsons = glob.glob(f'{SRC_DIR}/*.json')
imgs = glob.glob(f'{SRC_DIR}/*.png')

dst_img_dir = os.path.join(DST_DIR, "images")
os.path.isdir(dst_img_dir) or os.makedirs(dst_img_dir)
dst_label_dir = os.path.join(DST_DIR, "labels")
os.path.isdir(dst_label_dir) or os.makedirs(dst_label_dir)


dst_imgs = []
for i in range(len(jsons)):
    json_file = jsons[i]
    img_file = imgs[i]
    j = json.load(open(json_file, 'r'))

    dst_txt_path = os.path.join(dst_label_dir, os.path.splitext(os.path.basename(json_file))[0] + ".txt")
    dst_img_path = os.path.join(dst_img_dir, os.path.basename(img_file))

    os.path.isfile(dst_img_path) or shutil.copy(img_file, dst_img_path)

    img = cv2.imread(img_file)
    height, width, _ = img.shape

    dst_imgs.append(dst_img_path)
    with open(dst_txt_path, 'w') as f:
        for shape in j["shapes"]:
            index = LABELS.index(shape["label"])

            pts = ""
            for point in shape["points"]:
                pts += f'{point[0] / width} {point[1] / height} '

            line = f'{index} {pts}\n'
            f.write(line.strip())

with open(os.path.join(DST_DIR, "dataset.txt"), 'w') as f:
    for img in dst_imgs:
        f.write(img + "\n")