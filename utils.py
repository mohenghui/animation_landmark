import os
import platform
import base64
import json
IMG_TAIL=".png"
def makedirR(c_path,is_dir=True):
    if is_dir  and not os.path.exists(c_path):
        os.mkdir(c_path)
    elif not is_dir and not os.path.exists(c_path): #文件新建上一级目录
        if platform.system().lower() == 'windows':
            tmp='\\'.join(c_path.split('\\')[:-1])
        elif platform.system().lower() == 'linux':
            tmp='/'.join(c_path.split('/')[:-1])
        if  not os.path.exists(tmp):
            os.mkdir(tmp)
def imagedecode(j_path,save_path):
    with open(j_path, "r") as json_file:
        raw_data = json.load(json_file)
    image_base64_string = raw_data["imageData"]
    # 将 base64 字符串解码成图片字节码
    image_data = base64.b64decode(image_base64_string)
    # 将字节码以二进制形式存入图片文件中，注意 'wb'
    file_path,file_name=os.path.split(j_path)
    save_path=os.path.join(save_path,os.path.splitext(file_name)[0]+IMG_TAIL)
    with open(save_path, 'wb') as jpg_file:
        jpg_file.write(image_data)