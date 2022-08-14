from cProfile import label
from cmath import inf
import os
import json
from config import SplitData, SaveImage, DrawCricle,GetGT
import numpy as np
import cv2
from sympy.geometry import ( Line, Point)
from headpose import run
from utils import cal_point, imagedecode, makedirR, beint, listint,cal_distance
if __name__ == '__main__':
    # makedirR(GetGT.predict_dir)
    makedirR(GetGT.save_smilar)
    annotation_list = os.listdir(SplitData.annotation_path)
    img_list = os.listdir(GetGT.predict_dir)
    annotation_tail = ".json"
    for idx, img in enumerate(img_list):
        img_path = os.path.join(GetGT.predict_dir, img)
        og_img = cv2.imread(img_path)
        img_dir, img_name = os.path.split(img_path)
        annotation_path = os.path.join(
            SplitData.annotation_path, os.path.splitext(img_name)[0]+annotation_tail)
        print(annotation_path)
        if not os.path.exists(annotation_path):
            continue
        with open(annotation_path, "r") as f:
            row_data = json.load(f)
        # 读取每一条json数据
        landmark_info = []
        for d in row_data:
            if d == "shapes":
                for dd in row_data["shapes"]:
                    if dd["label"] not in landmark_info:
                        landmark_info.append(dd["points"][0])
        cv2.circle(og_img, beint(
            landmark_info[2]), 5, (69, 177, 53), -1)  # circle
        cv2.imshow('img', og_img)
        
        cv2.imwrite(os.path.join(GetGT.save_smilar,img),og_img)
        cv2.waitKey(0)

