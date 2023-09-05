import numpy as np
 
 
# 双目相機參數
class stereoCamera(object):
    def __init__(self):
        # 左相機内參
        self.cam_matrix_left = np.array([[1499.641, 0, 1097.616],
                                                             [0., 1497.989, 772.371],
                                                              [0., 0., 1.]])
        # 右相機内參
        self.cam_matrix_right = np.array([[1494.855, 0, 1067.321],
                                                               [0., 1491.890, 777.983],
                                                               [0., 0., 1.]])
 
        # 左右相機畸變係數:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.1103, 0.0789, -0.0004, 0.0017, -0.0095]])
        self.distortion_r = np.array([[-0.1065, 0.0793, -0.0002,  -8.9263e-06, -0.0161]])
 
        # 旋轉矩陣
        self.R = np.array([[0.9939, 0.0165, 0.1081],
                                     [-0.0157, 0.9998, -0.0084],
                                     [-0.1082, 0.0067, 0.9940]])
 
        # 平移矩陣
        self.T = np.array([[-423.716], [2.561], [21.973]])
 
        # 主點列坐標的差
        self.doffs = 0.0
 
        # 上述内參是否已經經過立體校正
        self.isRectified = False
 
    def setMiddleBurryParams(self):
        self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
                                                            [0., 3997.684, 187.5],
                                                            [0., 0., 1.]])
        self.cam_matrix_right =  np.array([[3997.684, 0, 225.0],
                                                                [0., 3997.684, 187.5],
                                                                [0., 0., 1.]])
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        self.R = np.identity(3, dtype= np.float64)
        self.T = np.array([[-193.001], [0.0], [0.0]])
        self.doffs = 131.111
        self.isRectified = True
