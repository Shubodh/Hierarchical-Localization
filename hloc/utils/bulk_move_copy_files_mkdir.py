import os
from shutil import copy2
from pathlib import Path

print("NOTE: Call this from py3d repo")
#def copy_to_local_system():
#    room_ids = ["1", "3", "5", "7", "9"]
#    scene_types_given_non_redundant = ["ROI_with_QOI", "RRI_with_QRI", "ROI_and_ARRI_with_QOI_and_AQRI"] # There are 2 more but redundant: ROI_with_QOI_and_AQRI, ROI_and_ARRI_with_QOI
#    scene_types_given = ["ROI_with_QOI", "RRI_with_QRI", "ROI_and_ARRI_with_QOI_and_AQRI", "ROI_with_QOI_and_AQRI", "ROI_and_ARRI_with_QOI"] # There are 2 more but redundant: ROI_with_QOI_and_AQRI, ROI_and_ARRI_with_QOI
#    scene_types_todo = ["RRI_with_QOI", "ROI_with_QRI", "ROI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QOI", "RRI_with_QRI_and_AQRI", "RRI_and_ARRI_with_QRI_and_AQRI"]
#    scene_types_all = scene_types_given + scene_types_todo
#
#    for room_id in room_ids:
#        base_path_ada = "/scratch/saishubodh/InLoc_like_RIO10/sampling10/scene0" + room_id
#        print("check ip address if stopping")
#        base_path_local = "shubodh@10.11.0.18:/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/datasets/InLoc_like_RIO10/sampling10_subset/scene0" + room_id 
#
#        #for each_type in scene_types_all:
#        #for each_type in scene_types_todo:
#        for each_type in scene_types_given:
#            print("scp -r " + base_path_ada + "_" + each_type + "/database/cutouts/*00000*" + " " + base_path_local + "_" + each_type + "/database/cutouts/")
#            os.system("scp -r " + base_path_ada + "_" + each_type + "/database/cutouts/*00000*" + " " + base_path_local + "_" + each_type + "/database/cutouts/")
#
#            # 0000 : 4 zeros
#            print("scp -r " + base_path_ada + "_" + each_type + "/query/*0000*" + " " + base_path_local + "_" + each_type + "/query/")
#            os.system("scp -r " + base_path_ada + "_" + each_type + "/query/*0000*" + " " + base_path_local + "_" + each_type + "/query/")
#            # 00000 : 5 zeros
#            #print("scp -r " + base_path_ada + "_" + each_type + "/query/*00000*" + " " + base_path_local + "_" + each_type + "/query/")
#            #os.system("scp -r " + base_path_ada + "_" + each_type + "/query/*00000*" + " " + base_path_local + "_" + each_type + "/query/")
#
#def bulk_mkdir_and_copy_files():
#    #ROI, QOI, RRI, QRI, ARRI, AQRI = [],[],[],[],[],[]
#    room_ids = ["1", "3", "5", "7", "9"]
#
#    scene_types_given = ["ROI_with_QOI", "RRI_with_QRI", "ROI_and_ARRI_with_QOI_and_AQRI"] # There are 2 more but redundant: ROI_with_QOI_and_AQRI, ROI_and_ARRI_with_QOI
#    scene_types_todo = ["RRI_with_QOI", "ROI_with_QRI", "ROI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QOI", "RRI_with_QRI_and_AQRI", "RRI_and_ARRI_with_QRI_and_AQRI"]
#    #room_ids = ["4"]
#    for room_id in room_ids:
#        print(f"room_id: {room_id} \n")
#        base_path = "/scratch/saishubodh/InLoc_like_RIO10/sampling10/scene0" + room_id
#        for each_type in scene_types_todo:
#            os.system("mkdir -p " + base_path + "_" + each_type + "/database/cutouts/")
#            os.system("mkdir -p " + base_path + "_" + each_type + "/query/")
#
#        ROI_v =  Path(base_path + "_" + "ROI_with_QOI/database/cutouts/")#frame-*
#        QOI_v =  Path(base_path + "_" + "ROI_with_QOI/query/") #frame-*"
#        RRI_v =  Path(base_path + "_" + "RRI_with_QRI/database/cutouts/")#frame-*"
#        QRI_v =  Path(base_path + "_" + "RRI_with_QRI/query/")#frame-*"
#        ARRI_v = Path(base_path + "_" + "ROI_and_ARRI_with_QOI_and_AQRI/database/cutouts/")#places-*"
#        AQRI_v = Path(base_path + "_" + "ROI_and_ARRI_with_QOI_and_AQRI/query/")#places-*"
#        f_suffix = 'frame-*'
#        a_suffix = 'places-*'
#
#        a_type = "RRI_with_QOI" 
#        print("1 ", a_type)
#        for every_img in list(RRI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/database/cutouts/")) 
#        for every_img in list(QOI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/query/")) 
#
#        a_type = "ROI_with_QRI"
#        print("2 ",a_type)
#        for every_img in list(ROI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/database/cutouts/")) 
#        for every_img in list(QRI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/query/")) 
#
#        a_type = "ROI_and_ARRI_with_QRI" 
#        print("3 ",a_type)
#        for every_img in list(ROI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/database/cutouts/")) 
#        for every_img in list(ARRI_v.glob(a_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/database/cutouts/")) 
#        for every_img in list(QRI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/query/")) 
#
#        a_type = "RRI_and_ARRI_with_QRI" 
#        print("4 ",a_type)
#        for every_img in list(RRI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/database/cutouts/")) 
#        for every_img in list(ARRI_v.glob(a_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/database/cutouts/")) 
#        for every_img in list(QRI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/query/")) 
#
#        a_type = "RRI_and_ARRI_with_QOI" 
#        print("5 ",a_type)
#        for every_img in list(RRI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/database/cutouts/")) 
#        for every_img in list(ARRI_v.glob(a_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/database/cutouts/")) 
#        for every_img in list(QOI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/query/")) 
#
#        a_type = "RRI_with_QRI_and_AQRI" 
#        print("6 ",a_type)
#        for every_img in list(RRI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/database/cutouts/")) 
#        for every_img in list(QRI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/query/")) 
#        for every_img in list(AQRI_v.glob(a_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/query/")) 
#
#        a_type = "RRI_and_ARRI_with_QRI_and_AQRI"
#        print("7 ", a_type)
#        for every_img in list(RRI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/database/cutouts/")) 
#        for every_img in list(ARRI_v.glob(a_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/database/cutouts/")) 
#        for every_img in list(QRI_v.glob(f_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/query/")) 
#        for every_img in list(AQRI_v.glob(a_suffix)): copy2(every_img, Path(base_path + "_" + a_type + "/query/")) 
#
#        #a_type = "RRI_with_QOI" 
#        #print("cp -r " + RRI_v + " " + base_path + "_" + a_type + "/database/cutouts/")
#        #print("cp -r " + QOI_v + " " + base_path + "_" + a_type + "/query/")
#
#        #a_type = "ROI_with_QRI"
#        #print("cp -r " + ROI_v + " " + base_path + "_" + a_type + "/database/cutouts/")
#        #print("cp -r " + QRI_v + " " + base_path + "_" + a_type + "/query/")
#
#        #a_type = "ROI_and_ARRI_with_QRI" 
#        #print("cp -r " + ROI_v + " " + base_path + "_" + a_type + "/database/cutouts/")
#        #print("cp -r " + ARRI_v+ " " + base_path + "_" + a_type + "/database/cutouts/")
#        #print("cp -r " + QRI_v + " " + base_path + "_" + a_type + "/query/")
#
#        #a_type = "RRI_and_ARRI_with_QRI" 
#        #print("cp -r " + RRI_v + " " + base_path + "_" + a_type + "/database/cutouts/")
#        #print("cp -r " + ARRI_v+ " " + base_path + "_" + a_type + "/database/cutouts/")
#        #print("cp -r " + QRI_v + " " + base_path + "_" + a_type + "/query/")
#
#        #a_type = "RRI_and_ARRI_with_QOI" 
#        #print("cp -r " + RRI_v + " " + base_path + "_" + a_type + "/database/cutouts/")
#        #print("cp -r " + ARRI_v+ " " + base_path + "_" + a_type + "/database/cutouts/")
#        #print("cp -r " + QOI_v + " " + base_path + "_" + a_type + "/query/")
#
#        #a_type = "RRI_with_QRI_and_AQRI" 
#        #print("cp -r " + RRI_v + " " + base_path + "_" + a_type + "/database/cutouts/")
#        #print("cp -r " + QRI_v + " " + base_path + "_" + a_type + "/query/")
#        #print("cp -r " + AQRI_v+ " " + base_path + "_" + a_type + "/query/")
#
#        #a_type = "RRI_and_ARRI_with_QRI_and_AQRI"
#        #print("cp -r " + RRI_v + " " + base_path + "_" + a_type + "/database/cutouts/")
#        #print("cp -r " + ARRI_v+ " " + base_path + "_" + a_type + "/database/cutouts/")
#        #print("cp -r " + QRI_v + " " + base_path + "_" + a_type + "/query/")
#        #print("cp -r " + AQRI_v+ " " + base_path + "_" + a_type + "/query/")
#
#if __name__ == '__main__':
#    #bulk_mkdir_and_copy_files()
#    copy_to_local_system()
