from cProfile import label
from cmath import inf
import os
import json
from config import SplitData, SaveImage, DrawCricle
import numpy as np
import cv2
from sympy.geometry import ( Line, Point)
from headpose import run
from utils import cal_point, imagedecode, makedirR, beint, listint
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
        # print(landmark_info[2])
        # cv2.circle(og_img,tuple([int(i) for i in landmark_info[2]]),5,(255,136,167),-1)
        # print(landmark_info)
        mid_point = [int(landmark_info[0][0]+landmark_info[4][0]) //
                     2, int(landmark_info[0][1]+landmark_info[4][1])//2]
        cv2.circle(og_img, tuple(mid_point), 5,
                   (28, 0, 28), -1)  # center chin and nose
        # print(landmark_info[2])
        cv2.line(og_img, tuple(mid_point), beint(
            landmark_info[3]), (77, 22, 55), 3)
        # cv2.imshow("show",og_img)
        # cv2.waitKey(0)
        # headpose_points_3 = [[int(i) for i in landmark_info[4]], [int(
        #     i) for i in landmark_info[0]], [int(i) for i in landmark_info[1]]]
        # headpose_points_4 = [[int(i) for i in landmark_info[4]], [int(
        #     i) for i in landmark_info[0]], [int(i) for i in landmark_info[5]]
        #     , [int(i) for i in landmark_info[6]]]
        # headpose_points_6 = [[int(i) for i in landmark_info[4]], [int(
        #     i) for i in landmark_info[0]], [int(i) for i in landmark_info[5]]
        #     , [int(i) for i in landmark_info[6]], [int(i) for i in landmark_info[7]]
        #     , [int(i) for i in landmark_info[8]]]
        headpose_points_6 = [listint(landmark_info[4]), listint(
            landmark_info[0]), listint(landmark_info[5]), listint(landmark_info[6]), listint(landmark_info[7]), list(landmark_info[8])]
        # headpose_points_6 = [landmark_info[4], landmark_info[0],
        #                      landmark_info[5], landmark_info[6],  landmark_info[7],  landmark_info[8]]
        k1,t_point=run(headpose_points_6, img=og_img)
        tmp_y=beint(landmark_info[3])[1]-tuple(mid_point)[1]
        tmp_x=beint(landmark_info[3])[0]-tuple(mid_point)[0]
        if tmp_x==0:predict_point=beint(
            landmark_info[1])
        else:
            k2=-(tmp_y/tmp_x)
            if k1==inf:
                predict_point=k2*landmark_info[1][0]
                mid_point=(landmark_info[1][0],predict_point)
            else:
                p1 = Point(beint(landmark_info[1]))
                p3 = Point(beint(landmark_info[4]))
                p4 = Point(t_point)
                l2_1 = Line(p3, p4)
                line3=l2_1.parallel_line(p1) #== Line(Point(0, 0), Point(0, -1))
                # print(len(line3))
                point2=[listint(landmark_info[1]),list(line3.args[1])]
                # print(point2)
                point1=[mid_point,listint(landmark_info[3])]
                predict_point=cal_point(point1,point2)
        # print(predict_point)
        cv2.circle(og_img, beint(predict_point), 5,
                   (0, 0, 255), -1)
        cv2.imshow('img', og_img)
        cv2.waitKey(0)

