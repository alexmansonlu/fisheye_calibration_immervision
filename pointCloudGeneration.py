
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import stereoconfig
import open3d as o3d
 
 
# preprocessing
def preprocess(img1, img2):
    """
    Preprocesses two images by converting them to grayscale and equalizing their histograms.
    
    Args:
        img1 (numpy.ndarray): The first image to be preprocessed.
        img2 (numpy.ndarray): The second image to be preprocessed.
        
    Returns:
        numpy.ndarray: The preprocessed version of the first image.
        numpy.ndarray: The preprocessed version of the second image.
    """
    # RGB->GRAY
    if (img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if (img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
    # Equalize histograms test
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)
 
    return img1, img2
 
 
# undistortion
def undistortion(image, camera_matrix, dist_coeff):
    """
    Generate the undistorted version of an image using the camera matrix and distortion coefficients.

    Parameters:
        image (ndarray): The input image.
        camera_matrix (ndarray): The camera matrix used for undistortion.
        dist_coeff (ndarray): The distortion coefficients.

    Returns:
        ndarray: The undistorted image.
    """
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)
 
    return undistortion_image
 
 

# Get the remapping Matrix and distortion map
def getRectifyTransform(height, width, config):
    """
    Generate the rectify transform for stereo images.

    Parameters:
        height (int): The height of the images.
        width (int): The width of the images.
        config (object): The configuration object containing camera matrices, distortion coefficients, rotation matrix, and translation vector.

    Returns:
        tuple: A tuple containing the rectification maps for the left and right images, the disparity-to-depth mapping matrix, and region of interest for the rectified images.
    """
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T
 
    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)
 
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
 
    return map1x, map1y, map2x, map2y, Q
 
 
# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    """
    Rectifies two images using the provided mappings.

    Args:
        image1 (numpy.ndarray): The first input image.
        image2 (numpy.ndarray): The second input image.
        map1x (numpy.ndarray): The mapping for the first image along the x-axis.
        map1y (numpy.ndarray): The mapping for the first image along the y-axis.
        map2x (numpy.ndarray): The mapping for the second image along the x-axis.
        map2y (numpy.ndarray): The mapping for the second image along the y-axis.
    
    Returns:
        tuple: A tuple containing the rectified version of image1 and image2.
    """
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
 
    return rectifyed_img1, rectifyed_img2
 
 
# 立體校正檢驗 劃綫
def draw_line(image1, image2):
    """
    Draw lines on an images to check for rectified effect.

    Parameters:
        image1 (ndarray): The first input image.
        image2 (ndarray): The second input image.

    Returns:
        ndarray: The output image with a line drawn on it.
    """
    # 輸出有line的圖片
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
 
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2
 
    # 繪製等距離平行綫
    line_interval = 50  # 直綫間隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)
 
    return output
 
 
# 視察計算
# The SGBM method is an intensity-based approach and generates a dense and smooth disparity map for good 3D reconstruction
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM參數設置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
 
    blockSize = 3
    paraml = {'minDisparity': 0,
              'numDisparities': 128,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }
 
    # contructe SGBM system
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)
 
    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]
 
        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right
 
    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.
 
    return trueDisp_left, trueDisp_right
 
# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]
 
    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)
 
    points_ = np.hstack((points_1, points_2, points_3))
 
    return points_

 
def getDepthMapWithQ(disparityMap : np.ndarray, Q : np.ndarray) -> np.ndarray:
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
 
    return depthMap.astype(np.float32)
 
def getDepthMapWithConfig(disparityMap : np.ndarray, config : stereoconfig.stereoCamera) -> np.ndarray:
    fb = config.cam_matrix_left[0, 0] * (-config.T[0])
    doffs = config.doffs
    depthMap = np.divide(fb, disparityMap + doffs)
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
    reset_index2 = np.where(disparityMap < 0.0)
    depthMap[reset_index2] = 0
    return depthMap.astype(np.float32)
 
 
if __name__ == '__main__':
    iml = cv2.imread('Adirondack-perfect/im0.png', 1)  # left image
    imr = cv2.imread('Adirondack-perfect/im1.png', 1)  # right image
    if (iml is None) or (imr is None):
        print("Error: Images are empty, please check your image's path!")
        sys.exit(0)
    height, width = iml.shape[0:2]
 
    # Read the intrinsic and extrinsic parameters
    # We stored them in the config file
    config = stereoconfig.stereoCamera()
    config.setMiddleBurryParams()
    print(config.cam_matrix_left)
 
    # recitify
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
    print(Q)
 
    # 繪製等距離平行綫，檢查立體校正的效果
    line = draw_line(iml_rectified, imr_rectified)
    
    cv2.imshow('Rectified', line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./data/check_rectification.png', line)
 
    # Semi-Global Block Matching - SGBM 立體 匹配
    iml_processed, imr_processed= preprocess(iml_rectified, imr_rectified)  #preprocess to eliminate uneven light cast
    disp, _ = stereoMatchSGBM(iml_processed, imr_processed, False) #execute SGBM
    cv2.imshow('Rectified', line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./data/disparity.png', disp * 4)
 
    # calculate the depth map
    # depthMap = getDepthMapWithQ(disp, Q)
    depthMap = getDepthMapWithConfig(disp, config)
    minDepth = np.min(depthMap)
    maxDepth = np.max(depthMap)
    print(minDepth, maxDepth)
    depthMapVis = (255.0 *(depthMap - minDepth)) / (maxDepth - minDepth)
    depthMapVis = depthMapVis.astype(np.uint8)
    cv2.imshow("DepthMap", depthMapVis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./data/DepthMap.png', depthMapVis)
 
    # draw the point cloud
    colorImage = o3d.geometry.Image(iml)
    depthImage = o3d.geometry.Image(depthMap)
    rgbdImage = o3d.geometry.RGBDImage().create_from_color_and_depth(colorImage, depthImage, depth_scale=1000.0, depth_trunc=np.inf)
    intrinsics = o3d.camera.PinholeCameraIntrinsic() # maybe need to use the fisheye
    # fx = Q[2, 3]
    # fy = Q[2, 3]
    # cx = Q[0, 3]
    # cy = Q[1, 3]
    fx = config.cam_matrix_left[0, 0]
    fy = fx
    cx = config.cam_matrix_left[0, 2]
    cy = config.cam_matrix_left[1, 2]
    print(fx, fy, cx, cy)
    intrinsics.set_intrinsics(width, height, fx= fx, fy= fy, cx= cx, cy= cy)
    extrinsics = np.array([[1., 0., 0., 0.],
                                        [0., 1., 0., 0.],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    pointcloud = o3d.geometry.PointCloud().create_from_rgbd_image(rgbdImage, intrinsic=intrinsics, extrinsic=extrinsics)
    o3d.io.write_point_cloud("PointCloud.pcd", pointcloud=pointcloud)
    o3d.visualization.draw_geometries([pointcloud], width=720, height=480)
    sys.exit(0)
