import os
from config import SplitData,SaveImage,DrawCricle
from utils import makedirR,imagedecode
if __name__ == '__main__':
    tp_files = os.listdir(SplitData.root_path)
    img_tail = ['.png', '.jpg', '.bmp']
    annotation_tail = ['.json', '.xml']
    makedirR(SplitData.annotation_path)
    makedirR(SplitData.img_path)
    for file in tp_files:
        file_path = os.path.join(SplitData.root_path, file)
        if os.path.isdir(file_path):
            continue
        file_tail = os.path.splitext(file)[1].lower()
        if file_tail in img_tail:
            new_path = os.path.join(SplitData.img_path, file)
            os.rename(file_path, new_path)
        elif file_tail in annotation_tail:
            new_path = os.path.join(SplitData.annotation_path, file)
            os.rename(file_path, new_path)
    
    annotation_list=os.listdir(SplitData.annotation_path)
    img_list=os.listdir(SplitData.img_path)
    annotation_tail=".json"
    for idx,img in enumerate(img_list):
        img_path=os.path.join(SplitData.img_path,img)
        img_dir,img_name=os.path.split(img_path)
        annotation_path=os.path.join(SplitData.annotation_path,os.path.splitext(img_name)[0]+annotation_tail)
        print(annotation_path)
        if not os.path.exists(annotation_path):continue
        makedirR(SaveImage.save_path) #获取原图数据
        imagedecode(annotation_path,SaveImage.save_path)
