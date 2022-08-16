from cProfile import label
from cmath import inf
import os
import json
from config import SplitData, SaveImage, DrawCricle
import numpy as np
import cv2
from sympy.geometry import ( Line, Point)
from headpose import run
from utils import cal_distance, cal_point, imagedecode, makedirR, beint, listint,get_foot,sign_cal_distance,scale_distance
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
        
        cv2.line(og_img, beint(landmark_info[2]), beint( 
        landmark_info[1]), (77, 22, 55), 1)  #额头距离
        cv2.line(og_img, beint(landmark_info[1]), beint(
        landmark_info[4]), (77, 22, 55), 1) #眉心到鼻子
        cv2.line(og_img, beint(landmark_info[4]), beint(
        landmark_info[0]), (77, 122, 55), 1) #鼻子到下巴
        # cv2.putText(og_img, 'pitch:' + str(pitch), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255))
        forehead_distance=cal_distance(landmark_info[2],landmark_info[1])#额头距离
        brow_nose_distance=cal_distance(landmark_info[1],landmark_info[4])#眉心到鼻子
        nose_chin_distance=cal_distance(landmark_info[4],landmark_info[0])#下巴距离
        cv2.putText(og_img, 'forehead:' + str(forehead_distance), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255))
        cv2.putText(og_img, 'brow_nose:' + str(brow_nose_distance), (5, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255))
        cv2.putText(og_img, 'nose_chin:' + str(nose_chin_distance), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255))
        pitch_flag=-1 #0平时,1低头,2抬头
        if landmark_info[1][1]<landmark_info[2][1]:
            pitch_flag=2
        elif (nose_chin_distance>brow_nose_distance or scale_distance(nose_chin_distance,brow_nose_distance,0.9))and nose_chin_distance>=forehead_distance*5:
            pitch_flag=2
        elif (forehead_distance>brow_nose_distance or scale_distance(forehead_distance,brow_nose_distance,0.9))and forehead_distance>=nose_chin_distance*2.9:
            pitch_flag=1
        elif nose_chin_distance*0.8>=brow_nose_distance and nose_chin_distance*0.8>=forehead_distance:
            #抬头
            pitch_flag=2
        elif nose_chin_distance<=brow_nose_distance*0.8 and nose_chin_distance<=forehead_distance*0.8:
            #低头
            pitch_flag=1
        else:
            #平视
            pitch_flag=0
        if pitch_flag==0:
            cv2.putText(og_img, 'ping' , (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255))
        elif pitch_flag==1:
            cv2.putText(og_img, 'di' , (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255))
        elif pitch_flag==2:
            cv2.putText(og_img, 'tai', (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255))
        cv2.imshow('img', og_img)
        cv2.waitKey(0)

