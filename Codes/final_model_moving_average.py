# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:05:53 2020

@author: skans
"""

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import numpy as np
from statsmodels.tsa.ar_model import AR
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import warnings
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
from math import log
from math import exp
from scipy.stats import boxcox
import pandas as pd
from statsmodels.tsa.arima_model import ARIMAResults
from math import exp
from math import log
import numpy as np
warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)

#%%
data = pd.read_csv("county_level_KG (FULL DAY).csv")
#%%
data.columns = ['COUNTY', 'p','d','q','score']
#%%
data2 = pd.read_csv("County level enrolment  data.csv")
data2['Year'] = data2['SCHOOL YEAR'].apply(lambda x: x[0:4])
#%%
# load dataset
data_kg_full = data2[['COUNTY','Year','KG (FULL DAY)']]
#%%
data_kg_half = data2[['COUNTY','Year','KG (HALF DAY)']]
#%%
data_pk_half = data2[['COUNTY','Year','PK (HALF DAY)']]
data_pk_half.dropna(inplace = True)
data_pk_half = data_pk_half.fillna(0)
#%%
data_pk_full = data2[['COUNTY','Year','PK (FULL DAY)']]
data_pk_full.dropna(inplace = True)
data_pk_full = data_pk_full.fillna(0)
#%%
dataframe_list = []
for county in data_kg_full.COUNTY.unique():
    county1 = data2[data2.COUNTY == county]
    series = pd.Series(county1['KG (FULL DAY)'].to_list(), index=county1['Year'].to_list())# create lagged dataset
    X= series.values
    train, test = X[1:len(X)-7], X[len(X)-7:]
    model = AR(train)
    model_fit = model.fit()
    print('Lag: %s' % model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)
    # make predictions
    print(len(train))
    print(len(train)+len(test)-1)
    predictions_list = []
    initial_list = []
    new_list = X.tolist()
    print(new_list)
    initial_list.append(new_list[0])
    initial_list.extend(train.tolist())
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)+5, dynamic=False)
    for i in range(len(predictions)):
        print(predictions[i])
        initial_list.append(predictions[i].round(0))
    year_list = county1['Year'].to_list()
    year_list.extend([2020,2021,2022,2023,2024,2025])
    county_list = [county for i in range(len(initial_list))]
    local = pd.DataFrame(np.column_stack([county_list, year_list,initial_list]))
    dataframe_list.append(local) 
#%%
dataframe_list = []
for county in data_kg_full.COUNTY.unique():
    county1 = data2[data2.COUNTY == county]
    series = pd.Series(county1['KG (FULL DAY)'].to_list(), index=county1['Year'].to_list())# create lagged dataset
    X= series.values
    train = X
    model = AR(train)
    model_fit = model.fit()
    print('Lag: %s' % model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)
    # make predictions
    # print(len(train))
    # print(len(train)+len(test)-1)
    predictions_list = []
    initial_list = []
    new_list = X.tolist()
    print(new_list)
    initial_list.extend(train.tolist())
    predictions = model_fit.predict(start=len(train), end=len(train)+5, dynamic=False)
    for i in range(len(predictions)):
        print(predictions[i])
        initial_list.append(predictions[i].round(0))
    year_list = county1['Year'].to_list()
    year_list.extend([2020,2021,2022,2023,2024,2025])
    county_list = [county for i in range(len(initial_list))]
    local = pd.DataFrame(np.column_stack([county_list, year_list,initial_list]))
    dataframe_list.append(local) 
#%%
finalframe = pd.concat(dataframe_list) 
#%%
finalframe.columns = ["Region", "Year", "Enrollment"]
#%%
finalframe.to_csv("COUNTY_ENROLLMENTS_KG_FULL_moving_average_orignal_and_next_five_years.csv", index = False)
#%%
dataframe_list = []

for county in data_kg_half.COUNTY.unique():
    county1 = data2[data2.COUNTY == county]
    series = pd.Series(county1['KG (HALF DAY)'].to_list(), index=county1['Year'].to_list())# create lagged dataset
    X= series.values
    if X[-1] == 0:
        continue 
    else :
        train, test = X[1:len(X)-5], X[len(X)-5:]
        model = AR(train)
        model_fit = model.fit()
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        # make predictions
        print(len(train))
        print(len(train)+len(test)-1)
        predictions_list = []
        initial_list = []
        new_list = X.tolist()
        print(new_list)
        initial_list.append(new_list[0])
        initial_list.extend(train.tolist())
        predictions = model_fit.predict(start=len(train), end=len(train)+len(test)+5, dynamic=False)
        for i in range(len(predictions)):
            print(predictions[i])
            initial_list.append(predictions[i].round(0))
        year_list = county1['Year'].to_list()
        year_list.extend([2020,2021,2022,2023,2024,2025])
        county_list = [county for i in range(len(initial_list))]
        local = pd.DataFrame(np.column_stack([county_list, year_list,initial_list]))
        dataframe_list.append(local) 
#%%
dataframe_list = []

for county in data_kg_half.COUNTY.unique():
    county1 = data2[data2.COUNTY == county]
    series = pd.Series(county1['KG (HALF DAY)'].to_list(), index=county1['Year'].to_list())# create lagged dataset
    X= series.values
    if X[-1] == 0:
        continue 
    else :
        train = X
        model = AR(train)
        model_fit = model.fit(trend = "nc")
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        # make predictions
        predictions_list = []
        initial_list = []
        new_list = X.tolist()
        print(new_list)
        initial_list.extend(train.tolist())
        predictions = model_fit.predict(start=len(X), end=len(X)+5, dynamic=False)
        for i in range(len(predictions)):
            print(predictions[i])
            initial_list.append(predictions[i].round(0))
        year_list = county1['Year'].to_list()
        year_list.extend([2020,2021,2022,2023,2024,2025])
        county_list = [county for i in range(len(initial_list))]
        local = pd.DataFrame(np.column_stack([county_list, year_list,initial_list]))
        dataframe_list.append(local) 
#%%
finalframe = pd.concat(dataframe_list) 
finalframe.columns = ["Region", "Year", "Enrollment"]
#%%
finalframe.to_csv("COUNTY_ENROLLMENTS_KG_HALF_moving_average_orignal_and_forecasted.csv", index = False)
#%%
dataframe_list = []
diff_counties = ["HERKIMER","QUEENS","FRANKLIN","ONEIDA","SCHENECTADY","JEFFERSON","SCHOHARIE"]
for county in data_pk_half.COUNTY.unique():
    county1 = data_pk_half[data_pk_half.COUNTY == county]
    series = pd.Series(county1['PK (HALF DAY)'].to_list(), index=county1['Year'].to_list())# create lagged dataset
    X= series.values
    if X[-1] == 0:
        continue  
    elif county in diff_counties:
        print(county)
        train, test = X[1:len(X)-1], X[len(X)-1:]
        model = AR(train)
        model_fit = model.fit(maxlag=1,trend = "nc")
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        # make predictions
        print(len(train))
        print(len(train)+len(test)-1)
        predictions_list = []
        initial_list = []
        new_list = X.tolist()
        print(new_list)
        initial_list.append(new_list[0])
        initial_list.extend(train.tolist())
        predictions = model_fit.predict(start=len(train), end=len(train)+len(test)+5, dynamic=False)
        for i in range(len(predictions)):
            print(predictions[i])
            initial_list.append(predictions[i].round(0))
        year_list = county1['Year'].to_list()
        year_list.extend([2020,2021,2022,2023,2024,2025])
        county_list = [county for i in range(len(initial_list))]
        local = pd.DataFrame(np.column_stack([county_list, year_list,initial_list]))
        dataframe_list.append(local) 
    else :
        print(county)
        train, test = X[1:len(X)-1], X[len(X)-1:]
        model = AR(train)
        model_fit = model.fit(maxlag=3, trend = "nc")
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        # make predictions
        print(len(train))
        print(len(train)+len(test)-1)
        predictions_list = []
        initial_list = []
        new_list = X.tolist()
        print(new_list)
        initial_list.append(new_list[0])
        initial_list.extend(train.tolist())
        predictions = model_fit.predict(start=len(train), end=len(train)+len(test)+5, dynamic=False)
        for i in range(len(predictions)):
            print(predictions[i])
            initial_list.append(predictions[i].round(0))
        year_list = county1['Year'].to_list()
        year_list.extend([2020,2021,2022,2023,2024,2025])
        county_list = [county for i in range(len(initial_list))]
        local = pd.DataFrame(np.column_stack([county_list, year_list,initial_list]))
        dataframe_list.append(local) 
#%%
dataframe_list = []
diff_counties = ["ULSTER","TIOGA","SCHUYLER","ONEIDA","SCHOHARIE"]
for county in data_pk_half.COUNTY.unique():
    county1 = data_pk_half[data_pk_half.COUNTY == county]
    series = pd.Series(county1['PK (HALF DAY)'].to_list(), index=county1['Year'].to_list())# create lagged dataset
    X= series.values
    if X[-1] == 0:
        continue  
    elif county in diff_counties:
        print(county)
        train = X
        model = AR(train)
        model_fit = model.fit(maxlag=1,trend = "nc")
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        # make predictions
        predictions_list = []
        initial_list = []
        new_list = X.tolist()
        print(new_list)
        initial_list.extend(train.tolist())
        predictions = model_fit.predict(start=len(train), end=len(train)+5, dynamic=False)
        for i in range(len(predictions)):
            print(predictions[i])
            initial_list.append(predictions[i].round(0))
        year_list = county1['Year'].to_list()
        year_list.extend([2020,2021,2022,2023,2024,2025])
        county_list = [county for i in range(len(initial_list))]
        local = pd.DataFrame(np.column_stack([county_list, year_list,initial_list]))
        dataframe_list.append(local) 
    else :
        print(county)
        train =X
        model = AR(train)
        model_fit = model.fit(trend = "nc")
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        # make predictions
        predictions_list = []
        initial_list = []
        new_list = X.tolist()
        print(new_list)
        initial_list.extend(train.tolist())
        predictions = model_fit.predict(start=len(X), end=len(X)+5, dynamic=False)
        for i in range(len(predictions)):
            print(predictions[i])
            initial_list.append(predictions[i].round(0))
        year_list = county1['Year'].to_list()
        year_list.extend([2020,2021,2022,2023,2024,2025])
        county_list = [county for i in range(len(initial_list))]
        local = pd.DataFrame(np.column_stack([county_list, year_list,initial_list]))
        dataframe_list.append(local)         
        #%%
finalframe = pd.concat(dataframe_list) 
finalframe.columns = ["Region", "Year", "Enrollment"]
#%%
finalframe.to_csv("COUNTY_ENROLLMENTS_PK_HALF_moving_average_orignal_and_forecasted.csv", index = False)
#%%
dataframe_list = []
diff_counties = ["HERKIMER","QUEENS","FRANKLIN","ONEIDA","SCHENECTADY","JEFFERSON","SCHOHARIE"]
for county in data_pk_full.COUNTY.unique():
    county1 = data_pk_full[data_pk_full.COUNTY == county]
    series = pd.Series(county1['PK (FULL DAY)'].to_list(), index=county1['Year'].to_list())# create lagged dataset
    X= series.values
    if X[-1] == 0:
        continue   
    else :
        print(county)
        train, test = X[1:len(X)-1], X[len(X)-1:]
        model = AR(train)
        model_fit = model.fit(maxlag=2, trend = "nc")
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        # make predictions
        print(len(train))
        print(len(train)+len(test)-1)
        predictions_list = []
        initial_list = []
        new_list = X.tolist()
        print(new_list)
        initial_list.append(new_list[0])
        initial_list.extend(train.tolist())
        predictions = model_fit.predict(start=len(train), end=len(train)+len(test)+5, dynamic=False)
        for i in range(len(predictions)):
            print(predictions[i])
            initial_list.append(predictions[i].round(0))
        year_list = county1['Year'].to_list()
        year_list.extend([2020,2021,2022,2023,2024,2025])
        county_list = [county for i in range(len(initial_list))]
        local = pd.DataFrame(np.column_stack([county_list, year_list,initial_list]))
        dataframe_list.append(local) 
#%%
dataframe_list = []
diff_counties = ["ULSTER","TIOGA","SCHUYLER","ONEIDA","SCHOHARIE"]
for county in data_pk_full.COUNTY.unique():
    county1 = data_pk_full[data_pk_full.COUNTY == county]
    series = pd.Series(county1['PK (FULL DAY)'].to_list(), index=county1['Year'].to_list())# create lagged dataset
    X= series.values
    if X[-1] == 0:
        continue   
    elif county in diff_counties:
        print(county)
        train = X
        model = AR(train)
        model_fit = model.fit(maxlag=2,trend = "nc")
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        # make predictions
        predictions_list = []
        initial_list = []
        new_list = X.tolist()
        print(new_list)
        initial_list.extend(train.tolist())
        predictions = model_fit.predict(start=len(train), end=len(train)+5, dynamic=False)
        for i in range(len(predictions)):
            print(predictions[i])
            initial_list.append(predictions[i].round(0))
        year_list = county1['Year'].to_list()
        year_list.extend([2020,2021,2022,2023,2024,2025])
        county_list = [county for i in range(len(initial_list))]
        local = pd.DataFrame(np.column_stack([county_list, year_list,initial_list]))
        dataframe_list.append(local) 
    else :
        print(county)
        train = X
        model = AR(train)
        model_fit = model.fit(maxlag = 3, trend = "c")
        print('Lag: %s' % model_fit.k_ar)
        print('Coefficients: %s' % model_fit.params)
        # make predictions
        print(len(train))
        predictions_list = []
        initial_list = []
        new_list = X.tolist()
        print(new_list)
        initial_list.extend(train.tolist())
        predictions = model_fit.predict(start=len(X), end=len(X)+5, dynamic=False)
        for i in range(len(predictions)):
            print(predictions[i])
            initial_list.append(predictions[i].round(0))
        year_list = county1['Year'].to_list()
        year_list.extend([2020,2021,2022,2023,2024,2025])
        county_list = [county for i in range(len(initial_list))]
        local = pd.DataFrame(np.column_stack([county_list, year_list,initial_list]))
        dataframe_list.append(local)
#%%
finalframe = pd.concat(dataframe_list) 
finalframe.columns = ["Region", "Year", "Enrollment"]
#%%
finalframe.to_csv("COUNTY_ENROLLMENTS_PK__moving_average_Orignal_forecasted.csv", index = False)