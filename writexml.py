from cProfile import label
from cmath import inf
from curses.textpad import rectangle
import os
import json
from config import SplitData, SaveImage, DrawCricle
import numpy as np
import cv2
from os import getcwd
from xml.etree import ElementTree as ET
from sympy.geometry import (Line, Point)
from headpose import run
from utils import cal_point, imagedecode, makedirR, beint, listint, cal_distance


class WriterXml:
    def __init__(self):
        self.detect_class = ["circle"]
        self.xml_annotation_path = "./data/annotations_xml"
        self.objectList=[]
        self.tail=".xml"
    def create_annotation(self, xn):
        global annotation
        tree = ET.ElementTree()
        tree.parse(xn)
        annotation = tree.getroot()

    # 遍历xml里面每个object的值如果相同就不插入
    def traverse_object(self, AnotPath):
        tree = ET.ElementTree(file=AnotPath)
        root = tree.getroot()
        ObjectSet = root.findall('object')
        for Object in ObjectSet:
            ObjName = Object.find('name').text
            BndBox = Object.find('bndbox')
            x1 = int(BndBox.find('xmin').text)
            y1 = int(BndBox.find('ymin').text)
            x2 = int(BndBox.find('xmax').text)
            y2 = int(BndBox.find('ymax').text)
            self.objectList.append([x1, y1, x2, y2, ObjName])

    # 定义一个创建一级分支object的函数
    def create_object(self, root, objl):  # 参数依次，树根，xmin，ymin，xmax，ymax
        # 创建一级分支object
        _object = ET.SubElement(root, 'object')
        # 创建二级分支
        name = ET.SubElement(_object, 'name')
        # print(obj_name)
        name.text = str(objl[4])
        pose = ET.SubElement(_object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(_object, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(_object, 'difficult')
        difficult.text = '0'
        # 创建bndbox
        bndbox = ET.SubElement(_object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = '%s' % objl[0]
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = '%s' % objl[1]
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = '%s' % objl[2]
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = '%s' % objl[3]

    # 创建xml文件的函数
    def create_tree(self, image_name, h, w, imgdir):
        global annotation
        # 创建树根annotation
        annotation = ET.Element('annotation')
        # 创建一级分支folder
        folder = ET.SubElement(annotation, 'folder')
        # 添加folder标签内容
        folder.text = (imgdir)

        # 创建一级分支filename
        filename = ET.SubElement(annotation, 'filename')
        filename.text = image_name

        # 创建一级分支path
        path = ET.SubElement(annotation, 'path')

        # path.text = getcwd() + '\{}\{}'.format(imgdir, image_name)  # 用于返回当前工作目录
        path.text = '{}/{}'.format(imgdir, image_name)  # 用于返回当前工作目录

        # 创建一级分支source
        source = ET.SubElement(annotation, 'source')
        # 创建source下的二级分支database
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'

        # 创建一级分支size
        size = ET.SubElement(annotation, 'size')
        # 创建size下的二级分支图像的宽、高及depth
        width = ET.SubElement(size, 'width')
        width.text = str(w)
        height = ET.SubElement(size, 'height')
        height.text = str(h)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'

        # 创建一级分支segmented
        segmented = ET.SubElement(annotation, 'segmented')
        segmented.text = '0'

    # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    def pretty_xml(self, element, indent, newline, level=0):
        if element:  # 判断element是否有子元素
            if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
                element.text = newline + indent * (level + 1)
            else:
                element.text = newline + indent * \
                    (level + 1) + element.text.strip() + \
                    newline + indent * (level + 1)
                # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
                # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
        temp = list(element)  # 将element转成list
        for subelement in temp:
            # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            if temp.index(subelement) < (len(temp) - 1):
                subelement.tail = newline + indent * (level + 1)
            else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
                subelement.tail = newline + indent * level
            self.pretty_xml(subelement, indent, newline,
                            level=level + 1)  # 对子元素进行递归操作
    def check(self):
        makedirR(self.xml_annotation_path)
        annotation_list = os.listdir(SplitData.annotation_path)
        img_list = os.listdir(SaveImage.save_path)
        annotation_tail = ".json"
        for idx, img in enumerate(img_list):
            img_path = os.path.join(SaveImage.save_path, img)
            og_img = cv2.imread(img_path)
            h, w = og_img.shape[:2]
            img_dir, img_name = os.path.split(img_path)
            annotation_path = os.path.join(
                SplitData.annotation_path, os.path.splitext(img_name)[0]+annotation_tail)
            print(annotation_path)
            if not os.path.exists(annotation_path):
                continue
            with open(annotation_path, "r") as f:
                row_data = json.load(f)
            # 读取每一条json数据
            # landmark_info = []
            circle_info = []
            for d in row_data:
                if d == "shapes":
                    for dd in row_data["shapes"]:
                        if dd["label"] == "3":
                            circle_info = dd["points"]
            if len(circle_info) == 0:
                continue
            cv2.circle(og_img, beint(
                circle_info[0]), 5, (69, 177, 53), -1)  # circle
            radius = cal_distance(circle_info[0], circle_info[1])
            cv2.circle(og_img, beint(
                circle_info[0]), int(radius), (69, 177, 53), 2)  # circle
            # print(radius)
            rectangle_left_point = [circle_info[0]
                                    [0]-radius, circle_info[0][1]-radius]
            rectangle_right_point = [circle_info[0]
                                     [0]+radius, circle_info[0][1]+radius]
            color=(0,0,255)

            cv2.rectangle(og_img,beint(rectangle_left_point),beint(rectangle_right_point), color, 2)
            cv2.imshow('img', og_img)
            cv2.waitKey(0)
    def work(self):
        makedirR(self.xml_annotation_path)
        annotation_list = os.listdir(SplitData.annotation_path)
        img_list = os.listdir(SaveImage.save_path)
        annotation_tail = ".json"
        for idx, img in enumerate(img_list):
            img_path = os.path.join(SaveImage.save_path, img)
            og_img = cv2.imread(img_path)
            h, w = og_img.shape[:2]
            img_dir, img_name = os.path.split(img_path)
            annotation_path = os.path.join(
                SplitData.annotation_path, os.path.splitext(img_name)[0]+annotation_tail)
            print(annotation_path)
            if not os.path.exists(annotation_path):
                continue
            with open(annotation_path, "r") as f:
                row_data = json.load(f)
            circle_info = []
            for d in row_data:
                if d == "shapes":
                    for dd in row_data["shapes"]:
                        if dd["label"] == "3":
                            circle_info = dd["points"]
            if len(circle_info) == 0:
                continue
            cv2.circle(og_img, beint(
                circle_info[0]), 5, (69, 177, 53), -1)  # circle
            radius = cal_distance(circle_info[0], circle_info[1])
            cv2.circle(og_img, beint(
                circle_info[0]), int(radius), (69, 177, 53), 2)  # circle
            rectangle_left_point = [circle_info[0]
                                    [0]-radius, circle_info[0][1]-radius]
            rectangle_right_point = [circle_info[0]
                                     [0]+radius, circle_info[0][1]+radius]

            xml_name = os.path.join(self.xml_annotation_path,os.path.splitext(img_name)[0]+self.tail)
            if (os.path.exists(xml_name)):
                self.create_annotation(xml_name)
                self.traverse_object(xml_name)
            else:
                self.create_tree(img_name, h, w, SaveImage.save_path)
            object_information = [int(rectangle_left_point[0]), int(
                rectangle_left_point[1]), int(rectangle_right_point[0]),int(rectangle_right_point[1]),self.detect_class[0]]
            if (self.objectList.count(object_information) == 0):
                                self.create_object(annotation, object_information)
            self.objectList = []
            # 将树模型写入xml文件
            tree = ET.ElementTree(annotation)
            root = tree.getroot()
            self.pretty_xml(root, '\t', '\n')
            tree.write(xml_name, encoding='utf-8')
if __name__ == '__main__':
    WriterXml().work() #运行
    # WriterXml().check() #检查
