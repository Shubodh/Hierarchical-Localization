import os

if __name__ == '__main__':
    room_ids = ["1", "3", "5", "7", "9"]
    dttime = "dt170322-t2240"
    for room_id in room_ids:
        print("\n")
        print(room_id)
        print("python3 io.py --pose_path /data/InLoc_dataset/outputs/rio/scene0"+ room_id  + "_AND_PLACES/scene0" + room_id  +"_sampling10_RIO_hloc_d2net-ss+NN-mutual_skip40_" + dttime +".txt" + " --scene_id 0" + room_id)
        os.system("python3 io.py --pose_path /data/InLoc_dataset/outputs/rio/scene0"+ room_id  + "_AND_PLACES/scene0" + room_id  +"_sampling10_RIO_hloc_d2net-ss+NN-mutual_skip40_" + dttime +".txt" + " --scene_id 0" + room_id)

