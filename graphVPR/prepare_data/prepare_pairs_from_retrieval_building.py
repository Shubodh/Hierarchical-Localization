import os

if __name__ == '__main__':
    # What this script does it: Input: Features, Output: Matching Pairs
    # Given a retrieval method like NetVLAD-top40, to be more precise, global descriptor of all images (.h5 file),
    # this script will call hloc/pairs_from_retrieval_custom.py to create best pairs (txt file)

    split_names = ['building_level_small_split1', 'building_level_small_split2']
    debug=False
    if debug==True:
        scene_names = [
        '8WUmhLawc2A'
        ]
        
        folder_names = [
        '0_mp3d_8WUmhLawc2A'
        ]
    topK_num = 40 
    retrieval_names = ["bruteforce", "hist-top3r-1i", "netvlad-top40", "netvlad-top3"]
    #input h5 file for netvlad-top40 or top3 would be same
    retrieval_name = retrieval_names[2]
    netvlad_h5_input = retrieval_names[2]
    for split in split_names:
        output_pairs_folder = "../../pairs/graphVPR/" +split+'/' +retrieval_name + "/"
        print(f"Outputting to folder: {output_pairs_folder}.txt")
        if not os.path.exists(output_pairs_folder):
            os.mkdir(output_pairs_folder)
        os.system("python ../../hloc/pairs_from_retrieval_custom.py"

        " --descriptors ../../outputs/graphVPR/" + split +'/'
        +netvlad_h5_input + "/global-feats-netvlad.h5"

        " --output " + output_pairs_folder  +  split + ".txt"
        
        " --num_matched " + str(topK_num)+ " --query_prefix query --db_prefix references")