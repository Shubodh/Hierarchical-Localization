import os

if __name__ == '__main__':
    room_ids = ["1","3", "5", "7", "9"]
    dt = "dt030422"
    #num_matched = "100"
    num_matched = "40"
    for room_id in room_ids:
        print("\n")
        print(room_id)
        print("python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene0"+ room_id  +"_JUST/global-feats-netvlad.h5 --num_matched "+ num_matched + " --output ../pairs/graphVPR/rio_metric/scene0"+room_id +"/netvlad"+num_matched+"_scene_JUST_sampling10_"+dt + ".txt --query_prefix query/ --db_prefix database/cutouts/")
        os.system("python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene0"+ room_id  +"_JUST/global-feats-netvlad.h5 --num_matched "+ num_matched + " --output ../pairs/graphVPR/rio_metric/scene0"+room_id +"/netvlad"+num_matched+"_scene_JUST_sampling10_"+dt + ".txt --query_prefix query/ --db_prefix database/cutouts/")

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
