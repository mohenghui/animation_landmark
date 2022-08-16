from cmath import inf
from dis import dis
import os
import platform
import base64
import json
from config import SplitData, SaveImage, DrawCricle
IMG_TAIL = ".png"


def makedirR(c_path, is_dir=True):
    if is_dir and not os.path.exists(c_path):
        os.mkdir(c_path)
    elif not is_dir and not os.path.exists(c_path):  # 文件新建上一级目录
        if platform.system().lower() == 'windows':
            tmp = '\\'.join(c_path.split('\\')[:-1])
        elif platform.system().lower() == 'linux':
            tmp = '/'.join(c_path.split('/')[:-1])
        if not os.path.exists(tmp):
            os.mkdir(tmp)


def imagedecode(j_path, save_path):
    with open(j_path, "r") as json_file:
        raw_data = json.load(json_file)
    image_base64_string = raw_data["imageData"]
    # 将 base64 字符串解码成图片字节码
    image_data = base64.b64decode(image_base64_string)
    # 将字节码以二进制形式存入图片文件中，注意 'wb'
    file_path, file_name = os.path.split(j_path)
    save_path = os.path.join(
        save_path, os.path.splitext(file_name)[0]+IMG_TAIL)
    with open(save_path, 'wb') as jpg_file:
        jpg_file.write(image_data)

def beint(tuple_list):
    return tuple([int(i) for i in tuple_list])
def listint(list):
    return [int(i)for i in list]
def cal_point(point1,point2):
    
    x1=point1[0][0]#取四点坐标
    y1=point1[0][1]
    x2=point1[1][0]
    y2=point1[1][1]
    
    x3=point2[0][0]
    y3=point2[0][1]
    x4=point2[1][0]
    y4=point2[1][1]
    
    k1=(y2-y1)*1.0/(x2-x1)#计算k1,由于点均为整数，需要进行浮点数转化
    b1=y1*1.0-x1*k1*1.0#整型转浮点型是关键
    if (x4-x3)==0:#L2直线斜率不存在操作
        k2=None
        b2=0
    else:
        k2=(y4-y3)*1.0/(x4-x3)#斜率存在操作
        b2=y3*1.0-x3*k2*1.0
    if k2==None:
        x=x3
    else:
        x=(b2-b1)*1.0/(k1-k2)
    y=k1*x*1.0+b1*1.0
    return [x,y]
def cal_distance(point1,point2):
    return pow(pow(point1[0]-point2[0],2)+pow(point1[1]-point2[1],2),1/2)


def get_foot(start_point, end_point, point_a):
    start_x, start_y = start_point
    end_x, end_y = end_point
    pa_x, pa_y = point_a

    p_foot = [0, 0]
    if start_point[0] == end_point[0]:
        p_foot[0] = start_point[0]
        p_foot[1] = point_a[1]
        return p_foot

    k = (end_y - start_y) * 1.0 / (end_x - start_x)
    a = k
    b = -1.0
    c = start_y - k * start_x
    p_foot[0] = int((b * b * pa_x - a * b * pa_y - a * c) / (a * a + b * b))
    p_foot[1] = int((a * a * pa_y - a * b * pa_x - b * c) / (a * a + b * b))

    return p_foot
def sign_cal_distance(point1,point2):
    if point1[1]<point2[1]:
        return pow(pow(point1[0]-point2[0],2)+pow(point1[1]-point2[1],2),1/2)
    else:
        return -pow(pow(point1[0]-point2[0],2)+pow(point1[1]-point2[1],2),1/2)

def scale_distance(distance1,distance2,scale):
    return True if distance1*scale<=distance2 or distance2*scale<=distance1 else False
    