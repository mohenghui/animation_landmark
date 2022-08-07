from cProfile import label
import os
import json
from config import SplitData,SaveImage,DrawCricle
import numpy as np
from  utils import imagedecode, makedirR
if __name__=='__main__':
    annotation_list=os.listdir(SplitData.annotation_path)
    img_list=os.listdir(SplitData.img_path)
    annotation_tail=".json"
    for idx,img in enumerate(img_list):
        img_path=os.path.join(SplitData.img_path,img)
        img_dir,img_name=os.path.split(img_path)
        annotation_path=os.path.join(SplitData.annotation_path,os.path.splitext(img_name)[0]+annotation_tail)
        print(annotation_path)
        if not os.path.exists(annotation_path):continue
        with open(annotation_path, "r") as f:
            row_data = json.load(f)
        # 读取每一条json数据
        landmark_info={}
        for d in row_data:
            if d == "shapes":
                for dd in row_data["shapes"]:
                    if dd["label"] not in landmark_info:
                        landmark_info[DrawCricle.classes[int(dd["label"])-1]]=dd["points"][0]
        print(landmark_info['chin'])
        break
        # makedirR(SaveImage.save_path) #获取原图数据
        # imagedecode(annotation_path,SaveImage.save_path)