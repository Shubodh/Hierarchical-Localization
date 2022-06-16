import os

if __name__ == '__main__':
    dt = "dt100622"
    time = "t0201"
    room_ids = ["1","3", "5", "7", "9"]
    #room_ids = ["9"]
    scene_types =["ROI_with_QOI" , "RRI_with_QRI" , "RRI_with_QOI" , "ROI_with_QRI"] # ["ROI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QRI", "ROI_and_ARRI_with_QOI"]  #more: ROI_with_QOI, RRI_with_QRI,
    scene_types_aug_ref = ["ROI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QRI","ROI_and_ARRI_with_QOI", "RRI_and_ARRI_with_QOI"]
    scene_types_aug_query = ["ROI_with_QOI_and_AQRI", "ROI_and_ARRI_with_QOI_and_AQRI", "RRI_with_QRI_and_AQRI", "RRI_and_ARRI_with_QRI_and_AQRI"]
    scene_types_aug_all = scene_types_aug_ref + scene_types_aug_query
    #scene_types_temp = ["ROI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QRI"]
    #scene_types_temp = [scene_types_aug_ref[2], scene_types_aug_ref[3], scene_types_aug_query[0],  scene_types_aug_query[1]]
    #scene_types_temp = [scene_types_aug_ref[2], scene_types_aug_ref[3]]#, scene_types_aug_query[0],  scene_types_aug_query[1]
    #scene_types_temp = [scene_types_aug_query[0],  scene_types_aug_query[1]]
    scene_types_temp = [scene_types_aug_query[2],  scene_types_aug_query[3]]
    #for scene_type in scene_types:
    #for scene_type in scene_types_aug_all:
    for scene_type in scene_types_temp:
        print("\n")
        print("\n")
        print(scene_type)
        for room_id in room_ids:
            print("\n")
            print(room_id)
            output_end ='scene0' + room_id + "_" + scene_type #'scene' + given_scene_id + '_and_places/' #'scene01_and_places' #'scene01_just/'

            print("python3 pipeline_InLoc_metric_level_ADA_RIO10-data_ALL.py --scene_id 0" + room_id +" --scene_type " +scene_type+" --date "+dt + " --time "+ time)
            os.system("python3 pipeline_InLoc_metric_level_ADA_RIO10-data_ALL.py --scene_id 0" + room_id +" --scene_type " +scene_type+" --date "+dt + " --time "+ time)
