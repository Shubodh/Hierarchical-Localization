import numpy as np
from utils import extract_RIO_instance_file, semantics_dict_from_set,frequency_list_to_dict, merge_all_rooms_dicts

def feat_vect(room_level_instance_files):
    '''Input: all instance files, say for N rooms.
    Output: Returns (N*num_objects) matrix, each row denoting room level histogram vector. 
    Just print variables commented out at end of this function to understand this function.'''
    # TODO: Current code does it for single inst file. You have to merge
    # sets of all rooms so that you have one vect of dim(objects from all rooms) instead from 1 room.
    num_rescan_rooms = len(room_level_instance_files)
    dict_instances_r = []
    for r_file in room_level_instance_files:
        dict_instances = extract_RIO_instance_file(r_file)
        dict_instances_r.append(dict_instances)

    oset_semantics, dict_semantics = merge_all_rooms_dicts(dict_instances_r)
    #print(dict_instances,'\n',oset_semantics,'\n',  dict_semantics)
    #print(len(dict_instances),'\n',len(oset_semantics),'\n',  len(dict_semantics))

    dict_freq_r = []
    for dict_instances in dict_instances_r:
        dict_freq = frequency_list_to_dict(list(dict_instances.values()))
        dict_freq_r.append(dict_freq)

    feature_dim = len(oset_semantics)
    featVect = np.zeros((num_rescan_rooms, feature_dim))

    for count, dict_freq in enumerate(dict_freq_r):
        for key in dict_freq:
            featVect[count, dict_semantics[key]] = dict_freq[key]

    #print(f"dict_semantics: \n {dict_semantics, len(dict_semantics)}")
    #print(f"dict_freq_r: \n {dict_freq_r}")
    #print(f"featVect: \n {featVect, featVect.shape}")

    return featVect, dict_semantics

def verify_featVect(featVect, room_level_instance_files, dict_semantics, rescan_rooms_ids):
    num_lines_r = []
    dict_instances_r = []
    for r_file in room_level_instance_files:
        dict_instances = extract_RIO_instance_file(r_file)
        dict_instances_r.append(dict_instances)
        num_lines_r.append(len(dict_instances.keys()))

    sumVect = np.sum(featVect, axis=1) 
    num_lines_r = np.array(num_lines_r)


    # 1st VERIFICATION
    debug_statement_1 = "VERIFICATION 1: \n We are comparing [no of instances in each instance.txt] \
to [summing featVect over room dimension], i.e. they've to be equal. \
If no of rooms is N, you should get N true's: (Uncomment code below for more info)"
    print(debug_statement_1)
    print(np.equal(sumVect, num_lines_r))
    #print(f"sumVect,num_lines_r : \n {sumVect, num_lines_r}")

    # 2nd VERIFICATION
    room_number = 1 # choose any of the rooms, between 0 to N-1
    print(f"\n VERIFICATION 2: Doing analysis for room_number {rescan_rooms_ids[room_number]}")
    debug_statement_2 = " You will see 3 objects below: \n \
1. dict_semantics: unique IDs for ALL the objects from all rooms \n \
2. feature vector for a particular room \n \
3. Given information from instances.txt file. \n \
OBSERVE and visually verify that feature vector is correct. \n"
    print(debug_statement_2)
    print(dict_semantics, "\n", featVect[room_number], "\n", dict_instances_r[room_number])