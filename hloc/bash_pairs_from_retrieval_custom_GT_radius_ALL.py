import os

if __name__ == '__main__':
    dt = "dt180622"
    #room_ids = ["1","3", "5", "7", "9"]
    room_ids = ["1"]
    scene_types =["ROI_with_QOI" , "RRI_with_QRI" , "RRI_with_QOI" , "ROI_with_QRI"] # ["ROI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QRI", "ROI_and_ARRI_with_QOI"]  #more: ROI_with_QOI, RRI_with_QRI,
    scene_types_aug_ref = ["ROI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QRI","ROI_and_ARRI_with_QOI", "RRI_and_ARRI_with_QOI"]
    scene_types_aug_query = ["ROI_with_QOI_and_AQRI", "ROI_and_ARRI_with_QOI_and_AQRI", "RRI_with_QRI_and_AQRI", "RRI_and_ARRI_with_QRI_and_AQRI"]
    scene_types_aug_all = scene_types_aug_ref + scene_types_aug_query
    #num_matched = "100"
    num_matched = "40"
    #current_scene_types = [scene_types_aug_ref[0]]
    current_scene_types = scene_types_aug_all
    for scene_type in current_scene_types:
        print("\n")
        print("\n")
        print("SCENE TYPE: ")
        print(scene_type)
        for room_id in room_ids:
            print("\n")
            print(room_id)
            output_end ='scene0' + room_id + "_" + scene_type #'scene' + given_scene_id + '_and_places/' #'scene01_and_places' #'scene01_just/'
            print("python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/"+ output_end +"/global-feats-netvlad.h5 --num_matched "+ num_matched + " --output ../pairs/graphVPR/rio_metric/scene0"+room_id +"/netvlad"+num_matched+"_scene_"+scene_type +"_sampling10_"+dt + ".txt --query_prefix query/ --db_prefix database/cutouts/")
            os.system("python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/"+ output_end +"/global-feats-netvlad.h5 --num_matched "+ num_matched + " --output ../pairs/graphVPR/rio_metric/scene0"+room_id +"/netvlad"+num_matched+"_scene_"+scene_type +"_sampling10_"+dt + ".txt --query_prefix query/ --db_prefix database/cutouts/")

#echo "scene 01"
#python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene01/global-feats-netvlad.h5 --num_matched 40 --output ../pairs/graphVPR/rio_metric/scene01/netvlad40_scene_sampling10_dt170322.txt --query_prefix query/ --db_prefix database/cutouts/
#echo "scene 03"
#python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene03/global-feats-netvlad.h5 --num_matched 40 --output ../pairs/graphVPR/rio_metric/scene03/netvlad40_scene_sampling10_dt170322.txt --query_prefix query/ --db_prefix database/cutouts/
#echo "scene 05"
#python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene05/global-feats-netvlad.h5 --num_matched 40 --output ../pairs/graphVPR/rio_metric/scene05/netvlad40_scene_sampling10_dt170322.txt --query_prefix query/ --db_prefix database/cutouts/
#echo "scene 07"
#python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene07/global-feats-netvlad.h5 --num_matched 40 --output ../pairs/graphVPR/rio_metric/scene07/netvlad40_scene_sampling10_dt170322.txt --query_prefix query/ --db_prefix database/cutouts/
#echo "scene 09"
#python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene09/global-feats-netvlad.h5 --num_matched 40 --output ../pairs/graphVPR/rio_metric/scene09/netvlad40_scene_sampling10_dt170322.txt --query_prefix query/ --db_prefix database/cutouts/
