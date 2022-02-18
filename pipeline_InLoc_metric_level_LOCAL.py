#!/usr/bin/env python
# coding: utf-8

# In[1]:

#
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

from pathlib import Path
from pprint import pformat
import sys

from hloc import extract_features, match_features, localize_inloc, visualization

import matplotlib.pyplot as plt


# In[2]:


# import hloc
# print("carefully inspect which hloc it is, whether the docker one or normal modified one.")
# hloc


# # Pipeline for indoor localization

# ## Setup
# Here we declare the paths to the dataset, image pairs, and we choose the feature extractor and the matcher. You need to download the [InLoc dataset](https://www.visuallocalization.net/datasets/) and put it in `datasets/inloc/`, or change the path.

# In[3]:


dataset = Path('datasets/inloc_small/')  # change this if your dataset is somewhere else

pairs = Path('pairs/inloc_small/')
loc_pairs = pairs / 'pairs-query-netvlad40-custom-shub-small-noDUC2.txt'  # top 40 retrieved by NetVLAD

outputs = Path('outputs/inloc_small/')  # where everything will be saved
results = outputs / 'InLoc_hloc_superpoint+superglue_netvlad40.txt'  # the result file


# In[4]:


# list the standard configurations available
# print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
# print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')


# In[5]:


# pick one of the configurations for extraction and matching
# you can also simply write your own here!
feature_conf = extract_features.confs['superpoint_inloc']
matcher_conf = match_features.confs['superglue']


# ## Extract local features for database and query images

# In[6]:


feature_path = extract_features.main(feature_conf, dataset, outputs)


# ## Match the query images
# Here we assume that the localization pairs are already computed using image retrieval (NetVLAD). To generate new pairs from your own global descriptors, have a look at `hloc/pairs_from_retrieval.py`. These pairs are also used for the localization - see below.

# In[7]:


match_path = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)


# ## Localize!
# Perform hierarchical localization using the precomputed retrieval and matches. Different from when localizing with Aachen, here we do not need a 3D SfM model here: the dataset already has 3D lidar scans. The file `InLoc_hloc_superpoint+superglue_netvlad40.txt` will contain the estimated query poses.

# In[ ]:


localize_inloc.main(
    dataset, loc_pairs, feature_path, match_path, results,
    skip_matches=20)  # skip database images with too few matches

# DO remember skip_matches is like k2, netvlad40 is k1 where k1 is 40. That means,
# k2 is further shortlisting these 40 images. (Following paper convention of PCLoc)


# ## Visualization
# We parse the localization logs and for each query image plot matches and inliers with a few database images.

# In[ ]:


visualization.visualize_loc(results, dataset, n=1, top_k_db=1, seed=2)
plt.show()


# In[ ]:




