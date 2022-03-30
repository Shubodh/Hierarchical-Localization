import os

if __name__ == '__main__':
    room_ids = ["3", "5"] #, "7", "9"]["1"]#,
    dt = "dt300322"
    #num_matched = "100"
    num_matched = "40"
    and_only_just ="_A-queryAND-ND_PLACES"#"_JU-queryAND-ST"# _AND_PLACES#_JUST
    for room_id in room_ids:
        print("\n")
        print(room_id)
        print("python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene0"+ room_id  +and_only_just+"/global-feats-netvlad.h5 --num_matched "+ num_matched + " --output ../pairs/graphVPR/rio_metric/scene0"+room_id +"/netvlad"+num_matched+"_scene"+and_only_just+"_sampling10_"+dt + ".txt --query_prefix query/ --db_prefix database/cutouts/")
        os.system("python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene0"+ room_id  +and_only_just+"/global-feats-netvlad.h5 --num_matched "+ num_matched + " --output ../pairs/graphVPR/rio_metric/scene0"+room_id +"/netvlad"+num_matched+"_scene"+and_only_just+"_sampling10_"+dt + ".txt --query_prefix query/ --db_prefix database/cutouts/")
