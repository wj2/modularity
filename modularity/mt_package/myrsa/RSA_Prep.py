#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:12:09 2022

@author: menghi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from scipy.spatial.distance import (pdist, squareform)
from scipy.stats import pearsonr, spearmanr
import math
from scipy import stats, linalg

def PrepareConditions(Spatial_Coordinates,Conceptual_Coordinates,data):
    counter = [0]
    spa_indexes = []
    spa_coordlist = []
    for i in Spatial_Coordinates:
        if np.sum(np.where(np.logical_and( np.logical_and(data['Seed'] == 1, data['SpatialCoordinates_1'] == i[1]), data['SpatialCoordinates_2']==i[0]))):
            coord = np.asarray(np.where(np.logical_and( np.logical_and(data['Seed'] == 1, data['SpatialCoordinates_1'] == i[1]), data['SpatialCoordinates_2']==i[0])))    
            spa_indexes.append(coord)
            spa_coordlist.append(i)
            counter =+1

    counter = [0]
    con_indexes = []
    con_coordlist = []
    for i in Conceptual_Coordinates:
        if np.sum(np.where(np.logical_and( np.logical_and(data['Seed'] == 2, data['ConceptualCoordinates_1'] == i[1]), data['ConceptualCoordinates_2']==i[0]))):
            coord = np.asarray(np.where(np.logical_and( np.logical_and(data['Seed'] == 2, data['ConceptualCoordinates_1'] == i[1]), data['ConceptualCoordinates_2']==i[0])))    
            con_indexes.append(coord)
            con_coordlist.append(i)
            counter =+1
    return spa_coordlist, spa_indexes, con_coordlist, con_indexes

def PrepareIrrelevantConditions(Spatial_Distractor,Conceptual_Distractor,data):
    counter = [0]
    irr_con_indexes = []
    irr_con_coordlist = []
    for i in Spatial_Distractor:
        if np.sum(np.where(np.logical_and( np.logical_and(data['Seed'] == 1, data['ConceptualCoordinates_1'] == i[1]), data['ConceptualCoordinates_2']==i[0]))):
            coord = np.asarray(np.where(np.logical_and( np.logical_and(data['Seed'] == 1, data['ConceptualCoordinates_1'] == i[1]), data['ConceptualCoordinates_2']==i[0])))    
            irr_con_indexes.append(coord)
            irr_con_coordlist.append(i)
            counter =+1

    counter = [0]
    irr_spa_indexes = []
    irr_spa_coordlist = []
    
    for i in Conceptual_Distractor:
        if np.sum(np.where(np.logical_and( np.logical_and(data['Seed'] == 2, data['SpatialCoordinates_1'] == i[1]), data['SpatialCoordinates_2']==i[0]))):
            coord = np.asarray(np.where(np.logical_and( np.logical_and(data['Seed'] == 2, data['SpatialCoordinates_1'] == i[1]), data['SpatialCoordinates_2']==i[0])))    
            irr_spa_indexes.append(coord)
            irr_spa_coordlist.append(i)
            counter =+1
    return irr_spa_coordlist, irr_spa_indexes, irr_con_coordlist, irr_con_indexes

def PrepareConditionsAddition(Spatial_Coordinates,Conceptual_Coordinates,data):
    counter = [0]
    spa_indexes = []
    spa_coordlist = []
    for i in Spatial_Coordinates:
        if np.sum(np.where(np.logical_and( data['Seed'] == 1, np.abs(100-(data['SpatialCoordinates_1']+data['SpatialCoordinates_2'])) == i))):
            coord = np.asarray(np.where(np.logical_and( data['Seed'] == 1, np.abs(100-(data['SpatialCoordinates_1']+data['SpatialCoordinates_2']))==i)))    
            spa_indexes.append(coord)
            spa_coordlist.append(i)
            counter =+1

    counter = [0]
    con_indexes = []
    con_coordlist = []
    for i in Conceptual_Coordinates:
        if np.sum(np.where(np.logical_and( data['Seed'] == 2,np.abs(100-( data['ConceptualCoordinates_1']+data['ConceptualCoordinates_2']))==i))):
            coord = np.asarray(np.where(np.logical_and( data['Seed'] == 2, np.abs(100-(data['ConceptualCoordinates_1']+ data['ConceptualCoordinates_2']))==i)))    
            con_indexes.append(coord)
            con_coordlist.append(i)
            counter =+1
    return spa_coordlist, spa_indexes, con_coordlist, con_indexes

def PrepareConditionsSubtraction(Spatial_Coordinates,Conceptual_Coordinates,data):
    counter = [0]
    spa_indexes = []
    spa_coordlist = []
    for i in Spatial_Coordinates:
        if np.sum(np.where(np.logical_and( data['Seed'] == 1, np.abs(data['SpatialCoordinates_1']-data['SpatialCoordinates_2']) == i))):
            coord = np.asarray(np.where(np.logical_and( data['Seed'] == 1, np.abs(data['SpatialCoordinates_1'] - data['SpatialCoordinates_2'])==i)))    
            spa_indexes.append(coord)
            spa_coordlist.append(i)
            counter =+1

    counter = [0]
    con_indexes = []
    con_coordlist = []
    for i in Conceptual_Coordinates:
        if np.sum(np.where(np.logical_and( data['Seed'] == 2, np.abs(data['ConceptualCoordinates_1']-data['ConceptualCoordinates_2'])==i))):
            coord = np.asarray(np.where(np.logical_and( data['Seed'] == 2, np.abs(data['ConceptualCoordinates_1']- data['ConceptualCoordinates_2'])==i)))    
            con_indexes.append(coord)
            con_coordlist.append(i)
            counter =+1
    return spa_coordlist, spa_indexes, con_coordlist, con_indexes

def FindNeighbours(ch_pos,idx,threshold):
    neigh = np.zeros(len(ch_pos))
    idx_ch = ch_pos[idx,:]
    for i in range(0,len(ch_pos)):
        dist = np.sqrt(np.sum((idx_ch-ch_pos[i])**2))
        neigh[i] = dist<threshold
        
    return neigh

def FindDataPoints(time, idx, radius):    
    TimePoints = np.argwhere(np.logical_and(time < time[idx]+radius, time > time[idx]-radius))
    return TimePoints

def AVG_T_S(data, neigh, TimePoints):
    row_sel = np.squeeze(np.array(np.where(neigh)).T)
    temp = data[row_sel,:]
    
    if np.logical_and(len(row_sel)>1,len(TimePoints)>1):
        data_avgd = np.average(temp[:,TimePoints],axis=1)
    elif np.logical_and(len(row_sel)>1,len(TimePoints)<2):
        data_avgd = temp
    elif np.logical_and(len(row_sel)<2,len(TimePoints)>1):
        data_avgd = print('One channel selected?')
    else:
        data_avgd = data
        print('What are you averaging?')
    
    return data_avgd

def MEG_Distance(MEG_Data, threshold, radius, time, ch_pos, ch_idx, time_idx):
    
    MEG_points = []#np.zeros(np.shape(MEG_Data)[0])
    
    neigh = FindNeighbours(ch_pos,ch_idx,threshold)
    TimePoints = FindDataPoints(time, time_idx, radius)
    
    for c in range(0,np.shape(MEG_Data)[0]):  
        MEG_points.append(np.squeeze(AVG_T_S(MEG_Data[c,:,:], neigh, TimePoints)))
    
    MEG_dist = pdist(MEG_points,'correlation')
    return MEG_dist

def Beh_MEG_Sim(MEG_dist,Beh_dist, metric):
    if metric=='pearson':
        r = pearsonr(MEG_dist,Beh_dist)[0]
    elif metric=='spearman':
        r = spearmanr(MEG_dist,Beh_dist)[0]
    else:
        r = print('Metric not available')
    return r


def computeRSA(time,ch_pos,MEG_Data,threshold,radius, Beh_dist, metric):
    
    Corr_Mat = np.zeros((np.shape(ch_pos)[0],len(time)))
    #Diss_Mat = np.zeros((np.shape(squareform(Beh_dist[:,0]))[0],np.shape(squareform(Beh_dist[:,0]))[0],np.shape(ch_pos)[0],len(time)))
    Diss_Mat = np.zeros((np.shape(squareform(Beh_dist))[0],np.shape(squareform(Beh_dist))[0],np.shape(ch_pos)[0],len(time)))
    #ScaledBeh = np.zeros((28,2,np.shape(ch_pos)[0],len(time)))
    #MEG_Fitted = np.zeros((28,2,np.shape(ch_pos)[0],len(time)))
    #Procrustes_Fit = np.zeros((np.shape(ch_pos)[0],len(time)))
    for t in range(0,np.shape(MEG_Data)[2]):
        print('timepoint processed ' + str(t))
        for c in range(0, np.shape(ch_pos)[0]):
            MEG_dist = MEG_Distance(MEG_Data, threshold, radius, time, ch_pos, c, t)
            r = Beh_MEG_Sim(MEG_dist,Beh_dist, metric)
            Corr_Mat[c,t]=r
            
            Diss_Mat[:,:,c,t] = squareform(MEG_dist)
            #import sklearn as sk
            #from sklearn import manifold
            #from scipy.spatial import procrustes
            
            #md_scaling = sk.manifold.MDS(n_components=2,dissimilarity='precomputed')

            #BEH_scaling = md_scaling.fit_transform(squareform(Beh_dist))
            #MEG_scaling = md_scaling.fit_transform(squareform(MEG_dist))

            #ScaledBeh[:,:,c,t], MEG_Fitted[:,:,c,t], Procrustes_Fit[c,t] = procrustes(BEH_scaling, MEG_scaling)

    return Corr_Mat, Diss_Mat
    

def computeRSA_ParCorr(time,ch_pos,MEG_Data,threshold,radius, Beh_dist, metric):
    
    Corr_Mat_2D = np.zeros((np.shape(ch_pos)[0],len(time)))
    Corr_Mat_1D = np.zeros((np.shape(ch_pos)[0],len(time)))
    Corr_Mat_1DF = np.zeros((np.shape(ch_pos)[0],len(time)))
    
    Diss_Mat = np.zeros((np.shape(squareform(Beh_dist[:,0]))[0],np.shape(squareform(Beh_dist[:,0]))[0],np.shape(ch_pos)[0],len(time)))
    #ScaledBeh = np.zeros((28,2,np.shape(ch_pos)[0],len(time)))
    #MEG_Fitted = np.zeros((28,2,np.shape(ch_pos)[0],len(time)))
    #Procrustes_Fit = np.zeros((np.shape(ch_pos)[0],len(time)))
    for t in range(0,np.shape(MEG_Data)[2]):
        print('timepoint processed ' + str(t))
        for c in range(0, np.shape(ch_pos)[0]):
            MEG_dist = MEG_Distance(MEG_Data, threshold, radius, time, ch_pos, c, t)
            
            C = np.zeros((np.shape(Beh_dist)[0],5))
            C[:,0] = np.squeeze(np.ones((np.shape(Beh_dist)[0],1)))
            C[:,1] = MEG_dist
            C[:,2:5] = Beh_dist
            corr = partial_corr(C, metric)
            Corr_Mat_2D[c,t] = corr[1,2]
            Corr_Mat_1D[c,t] = corr[1,3]
            Corr_Mat_1DF[c,t] = corr[1,4]
            
            Diss_Mat[:,:,c,t] = squareform(MEG_dist)
            #import sklearn as sk
            #from sklearn import manifold
            #from scipy.spatial import procrustes
            
            #md_scaling = sk.manifold.MDS(n_components=2,dissimilarity='precomputed')

            #BEH_scaling = md_scaling.fit_transform(squareform(Beh_dist))
            #MEG_scaling = md_scaling.fit_transform(squareform(MEG_dist))

            #ScaledBeh[:,:,c,t], MEG_Fitted[:,:,c,t], Procrustes_Fit[c,t] = procrustes(BEH_scaling, MEG_scaling)

    return Corr_Mat_2D,Corr_Mat_1D,Corr_Mat_1DF, Diss_Mat

def compute_ParrCorr_Abstract_RSA(time,ch_pos,Spa_MEG_Data,Con_MEG_Data,threshold,radius, Beh_dist, metric):
    
    Corr_Mat_2D = np.zeros((np.shape(ch_pos)[0],len(time)))
    Corr_Mat_1D = np.zeros((np.shape(ch_pos)[0],len(time)))
    Corr_Mat_1DF = np.zeros((np.shape(ch_pos)[0],len(time)))
    
    for t in range(0,np.shape(Spa_MEG_Data)[2]):
        print('timepoint processed ' + str(t))
        for c in range(0, np.shape(ch_pos)[0]):
            neigh = FindNeighbours(ch_pos,c,threshold)
            TimePoints = FindDataPoints(time, t, radius)
            
            SM = []
            CM = []
            for cond in range(0,np.shape(Spa_MEG_Data)[0]):
                SM.append(AVG_T_S(Spa_MEG_Data[cond,:,:], neigh, TimePoints))
            for cond in range(0,np.shape(Con_MEG_Data)[0]):    
                CM.append(AVG_T_S(Con_MEG_Data[cond,:,:], neigh, TimePoints))
            
            MEG_dist = dist_2_vecMEG(np.squeeze(SM), np.squeeze(CM))
            
            C = np.zeros((np.shape(Beh_dist)[0],5))
            C[:,0] = np.squeeze(np.ones((np.shape(Beh_dist)[0],1)))
            C[:,1] = MEG_dist
            C[:,2:5] = Beh_dist
            corr = partial_corr(C, metric)
            Corr_Mat_2D[c,t] = corr[1,2]
            Corr_Mat_1D[c,t] = corr[1,3]
            Corr_Mat_1DF[c,t] = corr[1,4]

            
    return Corr_Mat_2D,Corr_Mat_1D,Corr_Mat_1DF

def partial_corr(C, metric):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p))
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            if metric=='pearson':
                corr = pearsonr(res_i, res_j)[0]
            elif metric=='spearman':
                corr = spearmanr(res_i, res_j)[0]
            else:
                corr = print('Metric not available')
            
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr
def dist_2_vec(vec1,vec2):
    
    dist = np.zeros(np.shape(vec1)[0]*np.shape(vec2)[0])
    counter = 0
    if type(vec1) is list:
        
        for i in range(0,np.shape(vec1)[0]):
            for k in range(0,np.shape(vec2)[0]):            
                dist[counter] = math.dist(vec1[i],vec2[k])
                counter = counter+1
    else:
        
        for i in range(0,np.shape(vec1)[0]):
            for k in range(0,np.shape(vec2)[0]):            
                dist[counter] = math.dist(vec1[i,:],vec2[k,:])
                counter = counter+1
        
    return dist
    
def dist_2_vecMEG(vec1,vec2):
    
    #dist = np.zeros(np.shape(vec1)[0]*np.shape(vec2)[0])
    #counter = 0
    #if type(vec1) is list:
    #    
    #    for i in range(0,np.shape(vec1)[0]):
    #        for k in range(0,np.shape(vec2)[0]):            
    #            #dist[counter] = math.dist(vec1[i],vec2[k])
    #            r,p = spearmanr(vec1[i],vec2[k])
    #            dist[counter] = 1-r
    #            counter = counter+1
    #else:
    #    
    #    for i in range(0,np.shape(vec1)[0]):
    #        for k in range(0,np.shape(vec2)[0]):            
    #            r,p = spearmanr(vec1[i,:],vec2[k,:])
    #            dist[counter] = 1-r
    #            counter = counter+1
    corrmat,ps = spearmanr(vec1,vec2,axis=1)
    corrmat = 1-corrmat
    dist = np.matrix.flatten(corrmat[0:np.shape(vec1)[0],np.shape(vec1)[0]:])
    
        
    return dist
    

def compute_Abstract_RSA(time,ch_pos,Spa_MEG_Data,Con_MEG_Data,threshold,radius, Beh_dist, metric):
    
    RSA_Mat = np.zeros((np.shape(ch_pos)[0],len(time)))
    for t in range(0,np.shape(Spa_MEG_Data)[2]):
        print('timepoint processed ' + str(t))
        for c in range(0, np.shape(ch_pos)[0]):
            neigh = FindNeighbours(ch_pos,c,threshold)
            TimePoints = FindDataPoints(time, t, radius)
            
            SM = []
            CM = []
            for cond in range(0,np.shape(Spa_MEG_Data)[0]):
                SM.append(AVG_T_S(Spa_MEG_Data[cond,:,:], neigh, TimePoints))
            for cond in range(0,np.shape(Con_MEG_Data)[0]):    
                CM.append(AVG_T_S(Con_MEG_Data[cond,:,:], neigh, TimePoints))
            
            MEG_dist = dist_2_vecMEG(np.squeeze(SM), np.squeeze(CM))
            r = Beh_MEG_Sim(MEG_dist,Beh_dist, metric)
            RSA_Mat[c,t]=r
    return RSA_Mat
    
    