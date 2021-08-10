#!/usr/bin/env python
# coding: utf-8

# In[80]:


import h5py


# In[81]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, localize_inloc, visualization


# # Pipeline for indoor localization

# ## Setup
# Here we declare the paths to the dataset, image pairs, and we choose the feature extractor and the matcher. You need to download the [InLoc dataset](https://www.visuallocalization.net/datasets/) and put it in `datasets/inloc/`, or change the path.

# In[70]:


scene_name = 'i5noydFURQK' #8WUmhLawc2A, 'EDJbREhghzL', 'i5noydFURQK', 'jh4fc5c5qoQ', 'mJXqzFtmKg4'
# Set whether you want output for 2 queries or 2nd query or 1st query (small)
query_2 = '_only2query.txt'; queries_2 = '_2queries.txt'; small = '_small.txt'
h5_suffix = small

# change this if your dataset is somewhere else
dataset = Path('datasets/graphVPR/mp3d_' + scene_name + '_small/')
pairs = Path('pairs/graphVPR/mp3d_' + scene_name + '_small/')

outputs = Path('outputs/graphVPR/mp3d_' + scene_name + '_small/')  # where everything will be saved
# results = outputs / 'InLoc_hloc_superpoint+superglue_netvlad40.txt'  # the result file


# In[71]:


#loc_pairs = pairs / 'pairs-query-netvlad40.txt'  # top 40 retrieved by NetVLAD
loc_pairs = pairs / ('pairs-query-mp3d_' + scene_name + h5_suffix)  # top 40 retrieved by NetVLAD


# In[72]:


# list the standard configurations available
print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')


# In[73]:


# pick one of the configurations for extraction and matching
# you can also simply write your own here!
feature_conf = extract_features.confs['d2net-ss'] #superpoint_inloc
matcher_conf = match_features.confs['NN-mutual']
# matcher_conf = match_features.confs['superglue']


# ## Extract local features for database and query images

# In[74]:


feature_path = extract_features.main(feature_conf, dataset, outputs)


# In[78]:


feature_path


# In[94]:


# Experimentation
with h5py.File(feature_path, 'r') as hfile:
    for key in hfile.keys():
        print(key)
        dset = hfile["/mp3d_query/i5noydFURQK/1_rgb-i5noydFURQK-bathroom1.png"]
        for key2 in dset.keys():
            print(key2)
            print(dset[key2])
#         print(dset)
#         matches0 = dset['matches0']
#         m0_np = np.array(matches0)
#         print(f"m0_np.shape {m0_np.shape}")


# ## Match the query images
# Here we assume that the localization pairs are already computed using image retrieval (NetVLAD). To generate new pairs from your own global descriptors, have a look at `hloc/pairs_from_retrieval.py`. These pairs are also used for the localization - see below.

# In[76]:


match_path = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)


# In[77]:


match_path


# ## Localize!
# Perform hierarchical localization using the precomputed retrieval and matches. Different from when localizing with Aachen, here we do not need a 3D SfM model here: the dataset already has 3D lidar scans. The file `InLoc_hloc_superpoint+superglue_netvlad40.txt` will contain the estimated query poses.

# In[16]:


#localize_inloc.main(
#    dataset, loc_pairs, feature_path, match_path, results,
#    skip_matches=20)#20  # skip database images with too few matches


# ## Visualization
# We parse the localization logs and for each query image plot matches and inliers with a few database images.

# In[18]:


#visualization.visualize_loc(results, dataset, n=1, top_k_db=1, seed=2)


# In[ ]:




