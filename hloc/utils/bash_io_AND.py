import os

if __name__ == '__main__':
    room_ids = ["1", "3", "5", "7", "9"]
    #room_ids = ["1", "3", "5", "7"]
    dttime = "dt250322-t1535"
    netvlad_num ="100"# "40" #
    for room_id in room_ids:
        print("\n")
        print(room_id)
        print("python3 io.py --pose_path /data/InLoc_dataset/outputs/rio/scene0"+ room_id  + "_AND_PLACES/scene0" + room_id  +"_sampling10_netvlad" +netvlad_num  +"_RIO_hloc_d2net-ss+NN-mutual_skip40_" + dttime +".txt" + " --scene_id 0" + room_id)
        os.system("python3 io.py --pose_path /data/InLoc_dataset/outputs/rio/scene0"+ room_id  + "_AND_PLACES/scene0" + room_id  +"_sampling10_netvlad" +netvlad_num  +"_RIO_hloc_d2net-ss+NN-mutual_skip40_" + dttime +".txt" + " --scene_id 0" + room_id)
