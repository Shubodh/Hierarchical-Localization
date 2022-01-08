import random
import matplotlib.pyplot as plt
import PyQt5
import os

import cv2
import numpy as np
import json
import png
from scipy.spatial.distance import cdist
from tqdm import tqdm
from PIL import Image
from sklearn.feature_extraction.text import TfidfTransformer


def getMatchInds(ft_ref,ft_qry,topK=1,metric='cosine'):
    """
    This function takes two matrics and computes the distance between every vector in the matrices.
    For every query vector a ref. vector having shortest distance is retured
    """
    """
    metric: 'euclidean' or 'cosine' or 'correlation'
    """
    dMat = cdist(ft_ref,ft_qry,metric)
    mInds = np.argsort(dMat,axis=0)[:topK]        # shape: K x ft_qry.shape[0]
    return mInds


def convertBinary(vec):
    return np.where(vec < 1, 0, 1)

def getMatchIndsBinary(refer,query):

    """
    Takes input histogram vectors converts them to binary vectors and compute matched indices
    """
    refer_binary = convertBinary(refer)
    query_binary = convertBinary(query)
    query_ref_temp = np.zeros(np.shape(refer))
    mInds_bin_int = np.ones(np.shape(query)[0])*(-1) #initializing it with (-1)

    for j in range(np.shape(query)[0]):
        for i in range(np.shape(refer)[0]):
            query_ref_temp[i] = np.logical_and(refer_binary[i], query_binary[j]).astype(int)
            # The row with max no of 1's is our match:
            # NOTE: In case of multiple occurrences of the maximum values, 
            # the indices corresponding to the first occurrence are returned.
        mInds_bin_int[j] = np.argmax(np.sum(query_ref_temp,axis=1))
    return mInds_bin_int.reshape(1,-1)

def getMatchIndsBinaryTfidf(refer,query):

    """
    Takes input histogram vectors converts them to binary tf-idf vectors and compute matched indices
    """
    transformer = TfidfTransformer()
    refer_binary = convertBinary(refer)
    query_binary = convertBinary(query)
    tfidf_refbin = transformer.fit_transform(refer_binary)
    refer_bin = tfidf_refbin.toarray()
    query_bin = np.zeros(np.shape(query))
    for i in range(np.shape(query)[0]):
        query_bin[i] = transformer.fit_transform(np.reshape(query_binary[i], (1,-1))).toarray()

    mInds_bin = getMatchInds(refer_bin, query_bin)
    return mInds_bin

def getMatchIndsTfidf(refVect,qryVect,topK):

    """
    Takes input histogram vectors converts them to histogram tf-idf vectors and compute matched indices
    """

    transformer = TfidfTransformer()
    tfidf_ref = transformer.fit_transform(refVect)
    refer_new = tfidf_ref.toarray()
    #print("reference")
    #print(refer_new)
    query_new = np.zeros(np.shape(qryVect))
    for i in range(np.shape(qryVect)[0]):
        query_new[i] = transformer.fit_transform(np.reshape(qryVect[i], (1,-1))).toarray()
    #print("query")
    #print(query_new)

    mInds = getMatchInds(refer_new, query_new,topK)
    return mInds