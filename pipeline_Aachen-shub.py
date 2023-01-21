#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, pairs_from_covisibility
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization


# # Pipeline for outdoor day-night visual localization

# ## Setup
# Here we declare the paths to the dataset, the reconstruction and localization outputs, and we choose the feature extractor and the matcher. You only need to download the [Aachen Day-Night dataset](https://www.visuallocalization.net/datasets/) and put it in `datasets/aachen/`, or change the path.

# In[2]:


# dataset = Path('datasets/aachen/')  # change this if your dataset is somewhere else
dataset = Path('/data/aachen/')  # change this if your dataset is somewhere else
images = dataset / 'images/images_upright/'

pairs = Path('pairs/aachen/')
sfm_pairs = pairs / 'pairs-db-covis20.txt'  # top 20 most covisible in SIFT model
# loc_pairs = pairs / 'pairs-query-netvlad50.txt'  # top 50 retrieved by NetVLAD
loc_pairs = pairs / 'pairs-query-netvlad20-small.txt'  # top 50 retrieved by NetVLAD

outputs = Path('/data/InLoc_dataset/outputs/aachen/') # + output_end # where everything will be saved
#outputs = Path('outputs/aachen_small/')  # where everything will be saved
reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
results = outputs / 'Aachen_hloc_superpoint+superglue_netvlad50.txt'  # the result file


# In[3]:


# list the standard configurations available
# print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
# print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')


# In[4]:


# pick one of the configurations for extraction and matching
# you can also simply write your own here!
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']


# ## Extract local features for database and query images

# In[5]:


print("1. extract_features.main")
feature_path = extract_features.main(feature_conf, images, outputs)


# 

# The function returns the path of the file in which all the extracted features are stored.

# ## Generate pairs for the SfM reconstruction
# Instead of matching all database images exhaustively, we exploit the existing SIFT model to find which image pairs are the most covisible. We first convert the SIFT model from the NVM to the COLMAP format, and then do a covisiblity search, selecting the top 20 most covisibile neighbors for each image.

# In[6]:


print("2. colmap_from_nvm.main")
colmap_from_nvm.main(
    dataset / '3D-models/aachen_cvpr2018_db.nvm',
    dataset / '3D-models/database_intrinsics.txt',
    dataset / 'aachen.db',
    outputs / 'sfm_sift')


# In[7]:


print("3. pairs_from_covisibility.main")
pairs_from_covisibility.main(
    outputs / 'sfm_sift', sfm_pairs, num_matched=20)


# ## Match the database images

# In[8]:


print("4. match_features.main")
sfm_match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)


# The function returns the path of the file in which all the computed matches are stored.

# ## Triangulate a new SfM model from the given poses
# We triangulate the sparse 3D pointcloud given the matches and the reference poses stored in the SIFT COLMAP model.

# In[9]:


print("5. triangulation.main")
triangulation.main(
    reference_sfm,
    outputs / 'sfm_sift',
    images,
    sfm_pairs,
    feature_path,
    sfm_match_path,
    colmap_path='colmap')  # change if COLMAP is not in your PATH


# ## Match the query images
# Here we assume that the localization pairs are already computed using image retrieval (NetVLAD). To generate new pairs from your own global descriptors, have a look at `hloc/pairs_from_retrieval.py`. These pairs are also used for the localization - see below.

# In[10]:


print("6. match_features.main")
loc_match_path = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)


# ## Localize!
# Perform hierarchical localization using the precomputed retrieval and matches. The file `Aachen_hloc_superpoint+superglue_netvlad50.txt` will contain the estimated query poses. Have a look at `Aachen_hloc_superpoint+superglue_netvlad50.txt_logs.pkl` to analyze some statistics and find failure cases.

# In[11]:


print("6. localize_sfm.main")
localize_sfm.main(
    reference_sfm,
    dataset / 'queries/*_time_queries_with_intrinsics.txt',
    loc_pairs,
    feature_path,
    loc_match_path,
    results,
    covisibility_clustering=True)  # not required with SuperPoint+SuperGlue


# ## Visualizing the SfM model
# We visualize some of the database images with their detected keypoints.

# Color the keypoints by track length: red keypoints are observed many times, blue keypoints few.

# In[12]:


#visualization.visualize_sfm_2d(reference_sfm, images, n=1, color_by='track_length')
#
#
## Color the keypoints by visibility: blue if sucessfully triangulated, red if never matched.
#
## In[13]:
#
#
#visualization.visualize_sfm_2d(reference_sfm, images, n=1, color_by='visibility')
#
#
## Color the keypoints by triangulated depth: red keypoints are far away, blue keypoints are closer.
#
## In[14]:
#
#
#visualization.visualize_sfm_2d(reference_sfm, images, n=1, color_by='depth')
#
#
## ## Visualizing the localization
## We parse the localization logs and for each query image plot matches and inliers with a few database images.
#
## In[15]:
#
#
#visualization.visualize_loc(
#    results, images, reference_sfm, n=1, top_k_db=1, prefix='query/night', seed=2)
#
#
## In[ ]:
#
#
#
#
#
## In[ ]:
#
#
#
#
