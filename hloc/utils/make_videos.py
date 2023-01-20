import cv2
import os
import numpy as np

def make_videos():
    """
    Make videos from a directory of images. Take only all images with keyword "color".
    The input path where the images are located is "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene0X/seq0X/"
    where X is the scene number ranging from "01" to "10". There are subfolders in above path, make one video for each subfolder.
    The output path where the videos are saved is "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_viz_videos/scene0X/seq0X/" (create folders if not present).
    """
    input_path_small = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data_small/scene0X/seq0X/"
    input_path = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene0X/seq0X/"
    # input_path = input_path_small
    output_path = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_viz_videos/scene0X/"

    for i in range(1, 11):
        if i < 10:
            scene_num = "0" + str(i)
        else:
            scene_num = str(i)
        print("Scene: ", scene_num)
        scene_input_path = input_path.replace("0X", scene_num)
        scene_output_path = output_path.replace("0X", scene_num)
        if not os.path.exists(scene_output_path):
            os.makedirs(scene_output_path)

        for subfolder in os.listdir(scene_input_path):
            if os.path.isdir(scene_input_path + subfolder):
                subfolder_path = scene_input_path + subfolder + "/"
                print("Subfolder: ", subfolder_path)
                images = [img for img in os.listdir(subfolder_path) if img.endswith(".jpg") and "color" in img]
                images = [subfolder_path + img for img in images]
                images.sort()
                print("Num images: ", len(images))
                if len(images) > 0:
                    # print(images[0])
                    image = cv2.imread(images[0])
                    height, width, layers = image.shape
                    video = cv2.VideoWriter(scene_output_path + str(subfolder) + ".mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (width, height))
                    # video = cv2.VideoWriter(path=scene_output_path + str(subfolder) + ".mp4", fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps=30, frameSize=(width, height)
                    for image in images:
                        video.write(cv2.imread(image))
                    video.release()
                    print("Video saved to: ", scene_output_path + str(subfolder) + ".mp4")

make_videos()