import os
from config import SplitData
from utils import makedirR
if __name__=='__main__':
    tp_files=os.listdir(SplitData.root_path)
    img_tail=['.png','.jpg','.bmp']
    annotation_tail=['.json','.xml']
    makedirR(SplitData.annotation_path)
    makedirR(SplitData.img_path)
    for file in tp_files:
        file_path=os.path.join(SplitData.root_path,file)
        if os.path.isdir(file_path):continue
        file_tail=os.path.splitext(file)[1].lower()
        if file_tail in img_tail:
            new_path=os.path.join(SplitData.img_path,file)
            os.rename(file_path,new_path)
        elif file_tail in annotation_tail:
            new_path=os.path.join(SplitData.annotation_path,file)
            os.rename(file_path,new_path)
