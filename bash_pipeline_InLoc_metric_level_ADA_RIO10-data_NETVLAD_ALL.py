import os

if __name__ == '__main__':
    room_ids = ["1","3", "5", "7", "9"]
    #room_ids = ["1"]
    #room_ids = ["3", "5", "7", "9"]
    scene_types =["ROI_with_QOI" , "RRI_with_QRI" , "RRI_with_QOI" , "ROI_with_QRI"] # ["ROI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QRI", "ROI_and_ARRI_with_QOI"]  #more: ROI_with_QOI, RRI_with_QRI,
    scene_types_aug_ref = ["ROI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QRI","ROI_and_ARRI_with_QOI", "RRI_and_ARRI_with_QOI"]
    scene_types_aug_query = ["ROI_with_QOI_and_AQRI", "ROI_and_ARRI_with_QOI_and_AQRI", "RRI_with_QRI_and_AQRI", "RRI_and_ARRI_with_QRI_and_AQRI"]
    scene_types_aug_all = scene_types_aug_ref + scene_types_aug_query
    #current_scene_types = [scene_types_aug_ref[0]]
    current_scene_types = scene_types_aug_all
    #for scene_type in scene_types:
    for scene_type in current_scene_types:
        print("\n")
        print("\n")
        print("SCENE TYPE: ")
        print(scene_type)
        for room_id in room_ids:
            print("\n")
            print(f"room_id: {room_id}")
            print("python3 pipeline_InLoc_metric_level_ADA_RIO10-data_NETVLAD_ALL.py --scene_type " +  scene_type + " --scene_id 0" + room_id)
            os.system("python3 pipeline_InLoc_metric_level_ADA_RIO10-data_NETVLAD_ALL.py --scene_type " +  scene_type + " --scene_id 0" + room_id)
