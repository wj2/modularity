#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:13:42 2022

@author: menghi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne

def avg(data,idx):
    averaged = np.squeeze(np.mean(data[idx,:,:],axis=0))
    return averaged

def vis_erp(conds,time,legend):
    for i in conds:
        plt.plot(time,np.average(i,axis=0))
    plt.xlabel('Time')
    plt.ylabel('fT')
    plt.legend(legend)
    plt.show()

    return