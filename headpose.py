#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from __future__ import print_function
 
import os
import cv2
import sys
import numpy as np
import math
 
class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""
 
    def __init__(self, img_size=(480, 640)):
        self.size = img_size
 
        # 3D model points.
        self.model_points_6 = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=float) / 4.5
 
        self.model_points_14 = np.array([
             (6.825897, 6.760612, 4.402142),
             (1.330353, 7.122144, 6.903745),
             (-1.330353, 7.122144, 6.903745),
             (-6.825897, 6.760612, 4.402142),
             (5.311432, 5.485328, 3.987654),
             (1.789930, 5.393625, 4.413414),
             (-1.789930, 5.393625, 4.413414),
             (-5.311432, 5.485328, 3.987654),
             (2.005628, 1.409845, 6.165652),
             (-2.005628, 1.409845, 6.165652),
             (2.774015, -2.080775, 5.048531),
             (-2.774015, -2.080775, 5.048531),
             (0.000000, -3.116408, 6.097667),
             (0.000000, -7.415691, 4.070434)], dtype=float)
 
        self.model_points_68 = np.array([
            [-73.393523, -29.801432, -47.667532],
            [-72.775014, -10.949766, -45.909403],
            [-70.533638,   7.929818, -44.84258 ],
            [-66.850058,  26.07428 , -43.141114],
            [-59.790187,  42.56439 , -38.635298],
            [-48.368973,  56.48108 , -30.750622],
            [-34.121101,  67.246992, -18.456453],
            [-17.875411,  75.056892,  -3.609035],
            [  0.098749,  77.061286,   0.881698],
            [ 17.477031,  74.758448,  -5.181201],
            [ 32.648966,  66.929021, -19.176563],
            [ 46.372358,  56.311389, -30.77057 ],
            [ 57.34348 ,  42.419126, -37.628629],
            [ 64.388482,  25.45588 , -40.886309],
            [ 68.212038,   6.990805, -42.281449],
            [ 70.486405, -11.666193, -44.142567],
            [ 71.375822, -30.365191, -47.140426],
            [-61.119406, -49.361602, -14.254422],
            [-51.287588, -58.769795,  -7.268147],
            [-37.8048  , -61.996155,  -0.442051],
            [-24.022754, -61.033399,   6.606501],
            [-11.635713, -56.686759,  11.967398],
            [ 12.056636, -57.391033,  12.051204],
            [ 25.106256, -61.902186,   7.315098],
            [ 38.338588, -62.777713,   1.022953],
            [ 51.191007, -59.302347,  -5.349435],
            [ 60.053851, -50.190255, -11.615746],
            [  0.65394 , -42.19379 ,  13.380835],
            [  0.804809, -30.993721,  21.150853],
            [  0.992204, -19.944596,  29.284036],
            [  1.226783,  -8.414541,  36.94806 ],
            [-14.772472,   2.598255,  20.132003],
            [ -7.180239,   4.751589,  23.536684],
            [  0.55592 ,   6.5629  ,  25.944448],
            [  8.272499,   4.661005,  23.695741],
            [ 15.214351,   2.643046,  20.858157],
            [-46.04729 , -37.471411,  -7.037989],
            [-37.674688, -42.73051 ,  -3.021217],
            [-27.883856, -42.711517,  -1.353629],
            [-19.648268, -36.754742,   0.111088],
            [-28.272965, -35.134493,   0.147273],
            [-38.082418, -34.919043,  -1.476612],
            [ 19.265868, -37.032306,   0.665746],
            [ 27.894191, -43.342445,  -0.24766 ],
            [ 37.437529, -43.110822,  -1.696435],
            [ 45.170805, -38.086515,  -4.894163],
            [ 38.196454, -35.532024,  -0.282961],
            [ 28.764989, -35.484289,   1.172675],
            [-28.916267,  28.612716,   2.24031 ],
            [-17.533194,  22.172187,  15.934335],
            [ -6.68459 ,  19.029051,  22.611355],
            [  0.381001,  20.721118,  23.748437],
            [  8.375443,  19.03546 ,  22.721995],
            [ 18.876618,  22.394109,  15.610679],
            [ 28.794412,  28.079924,   3.217393],
            [ 19.057574,  36.298248,  14.987997],
            [  8.956375,  39.634575,  22.554245],
            [  0.381549,  40.395647,  23.591626],
            [ -7.428895,  39.836405,  22.406106],
            [-18.160634,  36.677899,  15.121907],
            [-24.37749 ,  28.677771,   4.785684],
            [ -6.897633,  25.475976,  20.893742],
            [  0.340663,  26.014269,  22.220479],
            [  8.444722,  25.326198,  21.02552 ],
            [ 24.474473,  28.323008,   5.712776],
            [  8.449166,  30.596216,  20.671489],
            [  0.205322,  31.408738,  21.90367 ],
            [ -7.198266,  30.844876,  20.328022]])
 
        self.focal_length = self.size[1] #w长度
        self.camera_center = (self.size[1] / 2, self.size[0] / 2) #相机中心
        self.camera_matrix = np.array(#相机矩阵
            [[self.focal_length, 0, self.camera_center[0]],#w,0,w_center
             [0, self.focal_length, self.camera_center[1]],#0,w,h_center
             [0, 0, 1]], dtype="double")
 
        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))
 
        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])
 
 
    def get_euler_angle(self, rotation_vector): #获得欧拉角
        # calc rotation angles
        theta = cv2.norm(rotation_vector, cv2.NORM_L2) #基准坐标 所有元素平方和的开平方
 
        # transform to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2)*rotation_vector[0][0] / theta
        y = math.sin(theta / 2)*rotation_vector[1][0] / theta
        z = math.sin(theta / 2)*rotation_vector[2][0] / theta
 
        # pitch (x-axis rotation)
        t0 = 2.0 * (w*x + y*z)
        t1 = 1.0 - 2.0*(x**2 + y**2)
        pitch = math.atan2(t0, t1)
 
        # yaw (y-axis rotation)
        t2 = 2.0 * (w*y - z*x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw = math.asin(t2)
 
        # roll (z-axis rotation)
        t3 = 2.0 * (w*z + x*y)
        t4 = 1.0 - 2.0*(y**2 + z**2)
        roll = math.atan2(t3, t4)
 
        return pitch, yaw, roll
 
    def solve_pose_by_6_points(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        points_6 = np.float32([
                    image_points[30], image_points[36], image_points[45],
                    image_points[48], image_points[54], image_points[8]])
 
        _, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_6, #三维坐标 ，float
            points_6, #二维坐标
            self.camera_matrix, #内参矩阵
            self.dist_coeefs,#相机畸变参数
            rvec=self.r_vec,#输出旋转向量
            tvec=self.t_vec,#输出平移矩阵
            useExtrinsicGuess=True)
 
        return rotation_vector, translation_vector
    
    def solve_pose_by_4_points(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        points_4 = np.float32([
                    image_points[0], image_points[36], image_points[45],
                    image_points[48]])

        _, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_6, #三维坐标 ，float
            points_4, #二维坐标
            self.camera_matrix, #内参矩阵
            self.dist_coeefs,#相机畸变参数
            rvec=self.r_vec,#输出旋转向量
            tvec=self.t_vec,#输出平移矩阵
            useExtrinsicGuess=True)

        return rotation_vector, translation_vector

    def solve_pose_by_14_points(self, image_points):
        points_14 = np.float32([
                        image_points[17], image_points[21], image_points[22], image_points[26], image_points[36],
                        image_points[39], image_points[42], image_points[45], image_points[31], image_points[35],
                        image_points[48], image_points[54], image_points[57], image_points[8]])
 
        _, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_14,
            points_14,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)
 
        return rotation_vector, translation_vector
 
    def solve_pose_by_68_points(self, image_points):
        _, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)
 
        return rotation_vector, translation_vector
 
    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))
 
        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
 
        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))
 
        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)
 
def run(pic_path):
    points_68 = load_anno(pic_path) #加载68个人脸特征点,自行实现
 
    img = cv2.imread(pic_path)
    pose_estimator = PoseEstimator(img_size=img.shape)
    pose = pose_estimator.solve_pose_by_6_points(points_68)
    #pose = pose_estimator.solve_pose_by_14_points(points_68)
    #pose = pose_estimator.solve_pose_by_68_points(points_68)
    pitch, yaw, roll = pose_estimator.get_euler_angle(pose[0]) #输入所有坐标信息
 
    def _radian2angle(r):
        return (r/math.pi)*180
 
    Y, X, Z = map(_radian2angle, [pitch, yaw, roll])
    line = 'Y:{:.1f}\nX:{:.1f}\nZ:{:.1f}'.format(Y,X,Z)
    print('{},{}'.format(os.path.basename(pic_path), line.replace('\n',',')))
 
    y = 20
    for _, txt in enumerate(line.split('\n')):
        cv2.putText(img, txt, (20, y), cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 1)
        y = y + 15
 
    for p in points_68:
        cv2.circle(img, (int(p[0]),int(p[1])), 2, (0,255,0), -1, 0)
 
    cv2.imshow('img', img)
    if cv2.waitKey(-1) == 27:
        pass
 
    return 0
 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("%(prog)s IMAGE_PATH")
        sys.exit(-1)
 
    sys.exit(run(sys.argv[1]))