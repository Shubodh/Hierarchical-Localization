import os


if __name__ == '__main__':
    #dttime = "dt030622-t1910"
    dt = "dt180622"
    time = "t1511"
    dttime = dt + "-" + time# "dt050622-t1111"
    room_ids = ["1", "3", "5", "7", "9"]
    #room_ids = ["1"]
    scene_types =["ROI_with_QOI" , "RRI_with_QRI" , "RRI_with_QOI" , "ROI_with_QRI"] # ["ROI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QRI", "ROI_and_ARRI_with_QOI"]  #more: ROI_with_QOI, RRI_with_QRI,
    scene_types_aug_ref = ["ROI_and_ARRI_with_QRI", "RRI_and_ARRI_with_QRI","ROI_and_ARRI_with_QOI", "RRI_and_ARRI_with_QOI"]
    scene_types_aug_query = ["ROI_with_QOI_and_AQRI", "ROI_and_ARRI_with_QOI_and_AQRI", "RRI_with_QRI_and_AQRI", "RRI_and_ARRI_with_QRI_and_AQRI"]
    scene_types_aug_all = scene_types_aug_ref + scene_types_aug_query
    netvlad_num = "40" # 100
    skip_no = "3" # "40" for d2net, 3 for SP
    sp_or_d2net = "superpoint_inloc+superglue" #"d2net-ss+NN-mutual"

    scene_types_temp = [scene_types[0],  scene_types_aug_ref[2]]
    #for scene_type in scene_types:
    for scene_type in scene_types_temp:
    #for scene_type in scene_types_aug_all:
        print("\n")
        print("\n")
        print("SCENE TYPE: ")
        print(scene_type)
        for room_id in room_ids:
            print("\n")
            print(room_id)
            output_end ='scene0' + room_id + "_" + scene_type #'scene' + given_scene_id + '_and_places/' #'scene01_and_places' #'scene01_just/'
            print("python3 io.py --pose_path /data/InLoc_dataset/outputs/rio/"+ output_end + "/" + scene_type + "_" + "GT_radius_scene0" + room_id  +"_sampling10_netvlad" +netvlad_num  +"_RIO_hloc_" + sp_or_d2net + "_skip"+ skip_no + "_" + dttime +".txt" + " --scene_id 0" + room_id)
            os.system("python3 io.py --pose_path /data/InLoc_dataset/outputs/rio/"+ output_end + "/" + scene_type + "_" + "GT_radius_scene0" + room_id  +"_sampling10_netvlad" +netvlad_num  +"_RIO_hloc_" + sp_or_d2net + "_skip"+ skip_no + "_" + dttime +".txt" + " --scene_id 0" + room_id)
