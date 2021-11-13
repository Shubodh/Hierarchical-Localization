import fileinput
import os
import sys
import shutil
from argparse import ArgumentParser

def make_unique(list_unique):
    ret_list = []
    for i in list_unique:
        if i not in ret_list:
            ret_list.append(i)
    return ret_list

# def make_queries_no_amb(duc_lines_imgs, duc_lines, amb_lines):
# 	print(duc_lines_imgs)
# 	print(duc_lines)
# 	amb_lines = [line.split('/')[-1] for line in amb_lines]
# 	print(amb_lines)
# 	print(len(amb_lines))
# 	amb_lines = make_unique(amb_lines)
# 	print(len(amb_lines))
def transfer_src_dest(src, dest):
    # print(src)
    # print(os.path.isfile(src))
    # print(dest)
    source_dest_dir = os.path.join(*dest.split('/')[:-1])
    print("DEBUG")
    #print(dest, source_dest_dir)
    #sys.exit()
    if not os.path.isdir(source_dest_dir):
        os.makedirs(source_dest_dir)

    shutil.copy(src, dest)

def make_queries_all(query_base_path, duc_lines_imgs, duc_lines):
    # print(duc_lines)
    # print(duc_lines_imgs)
    dict_duc = {}
    for i in duc_lines_imgs:
        dict_duc[i] = []

    for i in duc_lines_imgs:
        for j in duc_lines:
            if i == j.split('/')[-1]:
                dict_duc[i].append(j)
    print("Warning: these 2 found in two folders. Copying to first folder ONLY. PLEASE PROBE")
    for i in dict_duc.keys():
        if len(dict_duc[i]) > 1:
            print(dict_duc[i])
            dict_duc[i] = [dict_duc[i][0]]
    print("finally each query will be transfered as follows:")
    for i in dict_duc.keys():
        print(os.path.join(query_base_path+"iphone7",i) + '->' + os.path.join(query_base_path+"query_SG_GT",dict_duc[i][0]))
        transfer_src_dest(os.path.join(query_base_path+"iphone7",i), os.path.join(query_base_path+"query_SG_GT",dict_duc[i][0]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--query_folder', type=str, required=True, help='path to query folder, example ../../datasets/inloc/query/')
    args = parser.parse_args()
    query_base_path = args.query_folder
    
    duc_lines = []
    amb_lines = []
    for line in fileinput.input(files='gt_labels.txt'):
        if 'DUC' in line:
            duc_lines.append(line.strip())
        elif 'ambigious' not in line:
            amb_lines.append(line.strip())


    # print(amb_lines)
    # print(len(duc_lines))
    duc_lines = make_unique(duc_lines)
    duc_lines_imgs = [line.split('/')[-1] for line in duc_lines]
    print(len(duc_lines_imgs))
    #print(duc_lines_imgs)
    duc_lines_imgs = make_unique(duc_lines_imgs)

    #print(len(duc_lines_imgs))
    print(query_base_path)
    make_queries_all(query_base_path, duc_lines_imgs, duc_lines)
    #sys.exit()
    # make_queries_no_amb(duc_lines_imgs, duc_lines, amb_lines)