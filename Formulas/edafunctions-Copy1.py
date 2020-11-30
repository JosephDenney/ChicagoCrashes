import pandas as pd
import numpy as np 
import csv

from pltfunctions import hist_kde_plots
from haversine import haversine
from math import sqrt

import matplotlib.pyplot as plt
import seaborn as sns

def df_remove_columns_threshold(df):
    """
    purpose - to clean a dataframe
    input - dataframe
    output - dataframe less columns missing 5% or more of their data
    """
    dfthresh = len(df.index) - len(df.index)*0.05
    for column in df:
        if df[column].count() < dfthresh:
            df = df.drop(columns=column)
    return df

def df_merge_dataframes_left(df1, df2, merge_on):
    """
    purpose - to merge two dataframes
    input - two dataframes and the column they have in column
    output - single merged dataframe
    """
    data = pd.merge(df1, df2, how='left', on=merge_on)
    return data

def cost_benefit_analysis(model, X_test, y_test):
    y_preds = model.predict(X_test)
    label_dict = {"TP":0, "FP": 0, "TN": 0, "FN": 0}
    for yt, yp in zip(y_test, y_preds):
        if yt==yp:
            if yt==1:
                label_dict["TP"] += 1
            else:
                label_dict["TN"] += 1
        else:
            if yp==1:
                label_dict["FP"] += 1
            else:
                label_dict["FN"] += 1
    cb_dict = {"TP": 50, "FP": -10, "TN": 0, "FN": -60}
    total = 0
    for key in label_dict.keys():
        total += cb_dict[key]*label_dict[key]
    return total / sum(label_dict.values())
    
# def Precision():
    
# def Recall():