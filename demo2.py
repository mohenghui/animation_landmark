from cProfile import label
from cmath import inf
import os
import json
from config import SplitData, SaveImage, DrawCricle
import numpy as np
import cv2
from sympy.geometry import ( Line, Point)
from headpose import run
from utils import cal_point, imagedecode, makedirR, beint, listint,cal_distance
if __name__ == '__main__':
    makedirR(SaveImage.save_edit_path)
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
            landmark_info[0]), 5, (42, 140, 65), -1)  # chin
        cv2.circle(og_img, beint(
            landmark_info[1]), 5, (255, 136, 167), -1)  # eyebrow
        cv2.circle(og_img, beint(
            landmark_info[2]), 5, (69, 177, 53), -1)  # circle
        cv2.circle(og_img, beint(
            landmark_info[3]), 6, (50, 67, 128), -1)  # head
        cv2.circle(og_img, beint(
            landmark_info[4]), 5, (128, 0, 128), -1)  # nose
        cv2.circle(og_img, beint(
            landmark_info[5]), 5, (15, 67, 18), -1)  # eyeleft
        cv2.circle(og_img, beint(
            landmark_info[6]), 5, (48, 23, 78), -1)  # eyeright
        cv2.circle(og_img, beint(
            landmark_info[7]), 5, (15, 67, 18), -1)  # mouthleft
        cv2.circle(og_img, beint(
            landmark_info[8]), 5, (48, 23, 78), -1)  # mouthright
        cv2.line(og_img, beint(landmark_info[0]), beint(
        landmark_info[3]), (77, 22, 55), 3)
        # print((landmark_info[0][0]+landmark_info[3][0])*3/4)
        # distance=cal_distance(landmark_info[0],landmark_info[3])*2/3
        dis_x=(landmark_info[3][0]-landmark_info[0][0])*3/5
        dis_y=(landmark_info[3][1]-landmark_info[0][1])*3/5
        predict_point=[landmark_info[0][0]+dis_x,landmark_info[0][1]+dis_y]
        print(predict_point)
        # predict_point=[0,0]
        cv2.circle(og_img, beint(predict_point), 5,
                   (0, 0, 255), -1)
        cv2.imshow('img', og_img)
        
        cv2.imwrite(os.path.join(SaveImage.save_edit_path,img),og_img)
        cv2.waitKey(0)

