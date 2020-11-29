# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:05:30 2020

@author: skans
"""

import pandas as pd
import numpy as np
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pandas import Series
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
warnings.filterwarnings("ignore")
#%%
data = pd.read_csv("Conty_level_enrollment_Data.csv")
#%%
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) * 0.70)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in X]
	# make predictions
	predictions = list()
    
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    return best_cfg, best_score
# evaluate parameters

#%%
for cat in ['KG (FULL DAY)']:
    county_list = []
    score_list = []
    best_cfg_list = []
    for county in data.COUNTY.unique():
        countys = data[data.COUNTY == county]
        county1 = countys[countys[cat] > 0]
        series = pd.Series(county1[cat].to_list(), index=county1['Year'].to_list())
        p_values = range(0,5)
        d_values = range(0, 5)
        q_values = range(0, 4)
        best_cfg, best_score = evaluate_models(series.values, p_values, d_values, q_values)
        county_list.append(county)
        score_list.append(best_score)
        best_cfg_list.append(best_cfg)
    local = pd.DataFrame(np.column_stack([county_list,best_cfg_list,score_list]))
    name = "county_level_"+ cat+".csv"
    local.to_csv(name, index = False)
#%%
local = pd.DataFrame(np.column_stack([county_list,best_cfg_list,score_list]))
#%%
data = pd.read_csv("School_Charter_public.csv")
#%%
public = data[data['SCHOOL TYPE'] == "PUBLIC"]
charter = data[data['SCHOOL TYPE'] == "CHARTER"]
#%%
public['Year'] = public['SCHOOL YEAR'].apply(lambda x: x[0:4])
#%%
public = public.groupby(["Year","COUNTY"])['PK', 'PK (HALF DAY)',
       'PK (FULL DAY)', 'KG (HALF DAY)', 'KG (FULL DAY)'].sum().reset_index()
#%%
public.columns = ["Year","COUNTY",'PK', 'PK (HALF DAY)',
       'PK (FULL DAY)', 'KG (HALF DAY)', 'KG (FULL DAY)' ]
#%%
charter['Year'] = charter['SCHOOL YEAR'].apply(lambda x: x[0:4])

charter = charter.groupby(["Year","COUNTY"])['PK', 'PK (HALF DAY)',
       'PK (FULL DAY)', 'KG (HALF DAY)', 'KG (FULL DAY)'].sum().reset_index()
#%%
charter.columns = ["Year","COUNTY",'PK', 'PK (HALF DAY)',
       'PK (FULL DAY)', 'KG (HALF DAY)', 'KG (FULL DAY)' ]
#%%

for cat in ['PK', 'PK (HALF DAY)',
       'PK (FULL DAY)', 'KG (HALF DAY)', 'KG (FULL DAY)']:
    county_list = []
    score_list = []
    best_cfg_list = []
    for county in public.COUNTY.unique():
        countys = public[public.COUNTY == county]
        county1 = countys[countys[cat] > 0]
        series = pd.Series(county1[cat].to_list(), index=county1['Year'].to_list())
        p_values = range(0,5)
        d_values = range(0, 5)
        q_values = range(0, 4)
        best_cfg, best_score = evaluate_models(series.values, p_values, d_values, q_values)
        county_list.append(county)
        score_list.append(best_score)
        best_cfg_list.append(best_cfg)
    local = pd.DataFrame(np.column_stack([county_list,best_cfg_list,score_list]))
    name = "Enrollment/CharterSchool_"+ cat+".csv"
    local.to_csv(name, index = False)
#%%
local = pd.DataFrame(np.column_stack([county_list,best_cfg_list,score_list]))
#%%
data = pd.read_csv("School_Charter_public.csv")
#%%
public = data[data['SCHOOL TYPE'] == "PUBLIC"]
charter = data[data['SCHOOL TYPE'] == "CHARTER"]
#%%
public['Year'] = public['SCHOOL YEAR'].apply(lambda x: x[0:4])
#%%
public = public.groupby(["Year","COUNTY"])['PK', 'PK (HALF DAY)',
       'PK (FULL DAY)', 'KG (HALF DAY)', 'KG (FULL DAY)'].sum().reset_index()
#%%
public.columns = ["Year","COUNTY",'PK', 'PK (HALF DAY)',
       'PK (FULL DAY)', 'KG (HALF DAY)', 'KG (FULL DAY)' ]
#%%
charter['Year'] = charter['SCHOOL YEAR'].apply(lambda x: x[0:4])

charter = charter.groupby(["Year","COUNTY"])['PK', 'PK (HALF DAY)',
       'PK (FULL DAY)', 'KG (HALF DAY)', 'KG (FULL DAY)'].sum().reset_index()
#%%
charter.columns = ["Year","COUNTY",'PK', 'PK (HALF DAY)',
       'PK (FULL DAY)', 'KG (HALF DAY)', 'KG (FULL DAY)' ]
#%%

for cat in ['PK', 'PK (HALF DAY)',
       'PK (FULL DAY)', 'KG (HALF DAY)', 'KG (FULL DAY)']:
    county_list = []
    score_list = []
    best_cfg_list = []
    for county in public.COUNTY.unique():
        countys = public[public.COUNTY == county]
        county1 = countys[countys[cat] > 0]
        series = pd.Series(county1[cat].to_list(), index=county1['Year'].to_list())
        p_values = range(0,5)
        d_values = range(0, 5)
        q_values = range(0, 4)
        best_cfg, best_score = evaluate_models(series.values, p_values, d_values, q_values)
        county_list.append(county)
        score_list.append(best_score)
        best_cfg_list.append(best_cfg)
    local = pd.DataFrame(np.column_stack([county_list,best_cfg_list,score_list]))
    name = "Enrollment/CharterSchool_"+ cat+".csv"
    local.to_csv(name, index = False)
#%%
local = pd.DataFrame(np.column_stack([county_list,best_cfg_list,score_list]))
# #%%

