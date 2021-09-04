import os

if __name__ == '__main__':
    # What this script does it: Input: Features, Output: Matching Pairs
    # Given a retrieval method like NetVLAD-top40, to be more precise, global descriptor of all images (.h5 file),
    # this script will call hloc/pairs_from_retrieval_custom.py to create best pairs (txt file)

    scene_names = [
    '8WUmhLawc2A',
    'EDJbREhghzL',
    'i5noydFURQK',
    'jh4fc5c5qoQ',
    'mJXqzFtmKg4',
    'qoiz87JEwZ2',
    'RPmz2sHmrrY',
    'S9hNv5qa7GM',
    'ULsKaCPVFJR',
    'VzqfbhrpDEA',
    'wc2JMjhGNzB',
    'WYY7iVyf5p8',
    'X7HyMhZNoso',
    'YFuZgdQ5vWj',
    'yqstnuAEVhm'
    ]

    debug=False
    if debug==True:
        scene_names = [
        '8WUmhLawc2A'
        ]
        
        folder_names = [
        '0_mp3d_8WUmhLawc2A'
        ]
    topK_num = 3 
    retrieval_names = ["SP_SG_bruteforce", "hist-top3r-1i", "netvlad-top40", "netvlad-top3"]
    #input h5 file for netvlad-top40 or top3 would be same
    retrieval_name = retrieval_names[3]
    netvlad_h5_input = retrieval_names[2]
    for scene_name in scene_names:
        output_pairs_folder = "../../pairs/graphVPR/room_level_localization_small/" +retrieval_name + "/"
        print(f"Outputting to folder: {output_pairs_folder}{scene_name}.txt")
        if not os.path.exists(output_pairs_folder):
            os.mkdir(output_pairs_folder)
        os.system("python ../../hloc/pairs_from_retrieval_custom.py"

        " --descriptors ../../outputs/graphVPR/room_level_localization_small/"
        +netvlad_h5_input + "/"+scene_name+"/global-feats-netvlad.h5"

        " --output " + output_pairs_folder + scene_name + ".txt"
        
        " --num_matched " + str(topK_num)+ " --query_prefix query --db_prefix references")