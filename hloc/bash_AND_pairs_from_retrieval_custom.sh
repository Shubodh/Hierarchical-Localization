echo "scene 01"
python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene01_AND_PLACES/global-feats-netvlad.h5 --num_matched 40 --output ../pairs/graphVPR/rio_metric/scene01/netvlad40_scene_AND_PLACES_sampling10_dt170322.txt --query_prefix query/ --db_prefix database/cutouts/
echo "scene 03"
python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene03_AND_PLACES/global-feats-netvlad.h5 --num_matched 40 --output ../pairs/graphVPR/rio_metric/scene03/netvlad40_scene_AND_PLACES_sampling10_dt170322.txt --query_prefix query/ --db_prefix database/cutouts/
echo "scene 05"
python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene05_AND_PLACES/global-feats-netvlad.h5 --num_matched 40 --output ../pairs/graphVPR/rio_metric/scene05/netvlad40_scene_AND_PLACES_sampling10_dt170322.txt --query_prefix query/ --db_prefix database/cutouts/
echo "scene 07"
python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene07_AND_PLACES/global-feats-netvlad.h5 --num_matched 40 --output ../pairs/graphVPR/rio_metric/scene07/netvlad40_scene_AND_PLACES_sampling10_dt170322.txt --query_prefix query/ --db_prefix database/cutouts/
echo "scene 09"
python3 pairs_from_retrieval_custom.py --descriptors /data/InLoc_dataset/outputs/rio/scene09_AND_PLACES/global-feats-netvlad.h5 --num_matched 40 --output ../pairs/graphVPR/rio_metric/scene09/netvlad40_scene_AND_PLACES_sampling10_dt170322.txt --query_prefix query/ --db_prefix database/cutouts/
