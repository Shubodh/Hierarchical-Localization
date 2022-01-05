import open3d as o3d
import os
from pathlib import Path
import numpy as np
import cv2
import sys
import yaml

def camera_intrinsics(camera_params):
        fx, fy, cx, cy  = camera_params['camera_intrinsics']['model']
        width, height = camera_params['camera_intrinsics']['width'], camera_params['camera_intrinsics']['height']
        cam_int = o3d.camera.PinholeCameraIntrinsic(width,height,fx,fy,cx,cy)
        #print(width, height, fx, fy, cx, cy)
        #print("Camera intrinsics matrix:")
        #print(cam_int.intrinsic_matrix)
        return cam_int