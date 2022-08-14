from cProfile import label
from cmath import inf
import os
import json
from config import SplitData, SaveImage, DrawCricle
import numpy as np
import cv2
from sympy.geometry import ( Line, Point)
from headpose import run
from utils import cal_point, imagedecode, makedirR, beint, listint,get_foot,sign_cal_distance
if __name__ == '__main__':
    annotation_list = os.listdir(SplitData.annotation_path)
    img_list = os.listdir(SaveImage.save_path)
    annotation_tail = ".json"
    for idx, img in enumerate(img_list):
        img_path = os.path.join(SaveImage.save_path, img)
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
                        # landmark_info[DrawCricle.classes[int(dd["label"])-1]]=dd["points"][0]
                        landmark_info.append(dd["points"][0])
                        # print(tuple(landmark_info[-1]))
                        # cv2.circle(og_img,tuple([int(i) for i in landmark_info[-1]]),3,(255,0,0),-1)
        # print(landmark_info['chin'])
        cv2.circle(og_img, beint(
            landmark_info[0]), 3, (42, 140, 65), -1)  # chin
        cv2.circle(og_img, beint(
            landmark_info[1]), 3, (255, 136, 167), -1)  # eyebrow
        cv2.circle(og_img, beint(
            landmark_info[2]), 5, (69, 177, 53), -1)  # circle
        cv2.circle(og_img, beint(
            landmark_info[3]), 3, (50, 67, 128), -1)  # head
        cv2.circle(og_img, beint(
            landmark_info[4]), 3, (128, 0, 128), -1)  # nose
        cv2.namedWindow('img', cv2.WINDOW_FREERATIO)  
        
        brow_foot=get_foot(landmark_info[2],landmark_info[0],landmark_info[1])
        nose_foot=get_foot(landmark_info[2],landmark_info[0],landmark_info[4])
        
        cv2.circle(og_img, beint(
            brow_foot), 3, (145, 20, 108), -1)  # circle_to_browfoot
        cv2.circle(og_img, beint(
            nose_foot), 3, (148, 53, 178), -1)  # noisefoot_to_chin

        circle_to_browfoot=sign_cal_distance(landmark_info[2],brow_foot)
        noisefoot_to_chin=sign_cal_distance(nose_foot,landmark_info[0])
        cv2.line(og_img, beint(landmark_info[1]), beint(
        brow_foot), (77, 22, 55), 1)
        cv2.line(og_img, beint(landmark_info[2]), beint(
        brow_foot), (77, 22, 55), 1)
        cv2.line(og_img, beint(nose_foot), beint(
        landmark_info[4]), (77, 122, 55), 1)
        cv2.line(og_img, beint(nose_foot), beint(
        landmark_info[0]), (77, 122, 55), 1)
        max_distance=max(abs(circle_to_browfoot),abs(noisefoot_to_chin))
        print(circle_to_browfoot,noisefoot_to_chin)
        pitch =(circle_to_browfoot-noisefoot_to_chin)/max_distance
        cv2.putText(og_img, 'pitch:' + str(pitch), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255))
        cv2.imshow('img', og_img)
        cv2.waitKey(0)

