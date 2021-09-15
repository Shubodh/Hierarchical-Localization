import os
import json
import sys
from shutil import copyfile

if __name__ == '__main__':
    dataset_path = sys.argv[1]

    dict_scene = {}
    counter = 0
    dbow_path = os.path.join(dataset_path, "dbow")

    if not os.path.isdir(dbow_path):
        os.makedirs(dbow_path)

    image_path = os.path.join(dbow_path, "images")
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    # We'll add queries first
    query_path = os.path.join(dataset_path, "query")
    for room in os.listdir(query_path):
        room_path = os.path.join(query_path, room)
        for image in os.listdir(room_path):
            query = os.path.join(room_path, image)
            dest = os.path.join(image_path, "{}_rgb.png".format(counter))
            room_name = room.split('_')[-1]
            dict_scene[counter] = (room_name, image)
            copyfile(query, dest)
            counter += 1

    total_query_images = counter

    # We'll add ref images
    reference_path = os.path.join(dataset_path, "references")
    for room in os.listdir(reference_path):
        room_path = os.path.join(reference_path, room)
        for image in os.listdir(room_path):
            query = os.path.join(room_path, image)
            dest = os.path.join(image_path, "{}_rgb.png".format(counter))
            room_name = room.split('_')[-1]
            dict_scene[counter] = (room_name, image)
            copyfile(query, dest)
            counter += 1

    dict_save_path = os.path.join(dbow_path, "dictionary.json")
    dict_scene["num_queries"] = total_query_images
    json.dump(dict_scene, open(dict_save_path, 'w'))

    cmnd = "../..//DBoW2/build/./demo " + "/scratch/kanishanand/data_graphVPR/Hierarchical-Localization/" + image_path + \
        "/ " + str(total_query_images) + " " + str(counter)
    print(cmnd)
    os.system(cmnd)
