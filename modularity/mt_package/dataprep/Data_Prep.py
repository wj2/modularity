#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:28:15 2022

@author: menghi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import os

basefolder = "/Users/wjj/Dropbox/research/data/multi-tasking/"
bhv_folder = "Data_for_Jeff/Behavioural/"

def Load_Data(local, sbjN, training, test):
    if local==1:
        if np.logical_and(training == 0, test == 1):
            data = pd.read_csv(os.path.join(
                basefolder, 'PreTest/sbj', str(int(sbjN)), '_pretest.csv'
            ))
            events = np.load(os.path.join(basefolder, 'PreTest/PreTest_EventFile_sel.npy'))
            events_n = np.load(os.path.join(
                basefolder, 'PreTest/PreTest_EventFile_trls_sel.npy'
            ))
            epochs_cleaned = mne.read_epochs(
                os.path.join(basefolder, 'PreTest/PreTest_epochs_sel.fif')
            )
            
    elif local==0:
        if np.logical_and(training == 0, test == 1):
            data = pd.read_csv(
                os.path.join(basefolder, "PreTest/sbj{}_pretest.csv".format(sbjN))
            )
            events = np.load(os.path.join(basefolder, 'PreTest/PreTest_EventFile_sel.npy'))
            events_n = np.load(
                os.path.join(basefolder, 'PreTest/PreTest_EventFile_trls_sel.npy')
            )
            epochs_cleaned = mne.read_epochs(
                os.path.join(basefolder, "PreTest/PreTest_epochs_sel.fif")
            )
            
        elif np.logical_and(training == 0, test == 2):
            data = pd.read_csv(
                os.path.join(basefolder, "PostTest/sbj{}_posttest.csv".format(sbjN))
            )
            events = np.load(os.path.join(basefolder, 'PostTest/PostTest_EventFile_sel.npy'))
            events_n = np.load(
                os.path.join(basefolder, 'PostTest/PostTest_EventFile_trls_sel.npy')
            )
            epochs_cleaned = mne.read_epochs(
                os.path.join(basefolder, "PostTest/PostTest_epochs_sel.fif")
            )
            # data = pd.read_csv('/data/pt_02648/spatual/Behavioural/sbj'+sbjN+'/sbj'+str(int(sbjN))+'_posttest.csv')
            # events = np.load('/data/pt_02648/spatual/Preprocessed/sbj'+sbjN+'/PostTest/PostTest_EventFile_sel.npy')
            # events_n = np.load('/data/pt_02648/spatual/Preprocessed/sbj'+sbjN+'/PostTest/PostTest_EventFile_trls_sel.npy')
            # epochs_cleaned = mne.read_epochs('/data/pt_02648/spatual/Preprocessed/sbj'+sbjN+'/PostTest/PostTest_epochs_sel.fif')
        elif (training == 1):
            data = pd.read_csv(
                os.path.join(basefolder, "Training/sbj{}_training.csv".format(sbhN))
            )
            events = np.load(os.path.join(basefolder, 'Training/Training_EventFile_sel.npy'))
            events_n = np.load(
                os.path.join(basefolder, 'Training/Training_EventFile_trls_sel.npy')
            )
            epochs_cleaned = mne.read_epochs(
                os.path.join(basefolder, "Training/Training_epochs_sel.fif")
            )

            # data = pd.read_csv('/data/pt_02648/spatual/Behavioural/sbj'+sbjN+'/sbj'+str(int(sbjN))+'_training.csv')
            # events = np.load('/data/pt_02648/spatual/Preprocessed/sbj'+sbjN+'/Training/Training_EventFile_sel.npy')
            # events_n = np.load('/data/pt_02648/spatual/Preprocessed/sbj'+sbjN+'/Training/Training_EventFile_trls_sel.npy')
            # epochs_cleaned = mne.read_epochs('/data/pt_02648/spatual/Preprocessed/sbj'+sbjN+'/Training/Training_epochs_sel.fif')
    
    return epochs_cleaned, data, events_n, events

def Trial_Selection(data, trigger, events_n, epochs):
    trl_n = np.asarray(data['Trial_n']-1)

    idx = events_n[:,0]==trigger
    Rejected_trls = np.setxor1d(events_n[idx,1],trl_n)

    data = data.drop(Rejected_trls)
    data = data.reset_index()
    del data['index']

    Trig_Sel = np.asarray(np.where(events_n[:,0]==trigger))
    sel = Trig_Sel[:, data['Missed'] == 0]
    
    data = data[data.Missed != 1]
    #data = data.drop(np.squeeze(np.where(data['Missed']==1)))
    
    data = data.reset_index()
    del data['index']
    
    epochs_selected = epochs[np.squeeze(sel)]
    return data, epochs_selected
