# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:52:45 2020

@author: skans
"""

# evaluate ARIMA models with box-cox transformed time series
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
#%%
data = pd.read_csv("county_level_KG (FULL DAY).csv")
data.columns = ['COUNTY', 'p','d','q','score']
#%%
data2 = pd.read_csv("Conty_level_enrollment_Data.csv")
data2['Year'] = data2['SCHOOL YEAR'].apply(lambda x: x[0:4])

#%%
data.dropna(inplace = True)
#%%
dataframe_list = []
#%%
from math import log
from math import exp
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)

for index,row in data.iterrows():
    county = getattr(row, "COUNTY")
    p = getattr(row, "p")
    d = getattr(row, "d")
    q = getattr(row, "q")
    county1 = data2[data2.COUNTY == county]
    county1.fillna(0, inplace = True)
    county1.set_index("Year", inplace = True)
    value = county1.loc['2019','KG (FULL DAY)']
    if value > 0 :
        try:
            print(county)
            county1.reset_index(inplace = True)
            county1.Year = county1.Year.astype(int)
            county2 = county1[county1['KG (FULL DAY)'] > 0]
            series = pd.Series(county2['KG (FULL DAY)'].to_list(), index=county2.Year.to_list())
            X = series.values
            X = X.astype('float32')
            print(X)
            # train_size = int(len(X) * 0.80)
            # train, test = X[0:train_size], X[train_size:]
            # walk-forward validation
            history = [x for x in X]
            predictions = list()
            for i in range(1,12):
                transformed, lam = boxcox(history)
                if lam <-5:
                    transformed, lam = history , 1
                model = ARIMA(transformed, order = (int(p),int(d),int(q)))
                model_fit = model.fit(disp=0)
                yhat = model_fit.forecast()[0]
                yhat = boxcox_inverse(yhat, lam)
                yhat = np.round(yhat,0)
                predictions.append(yhat)
                history.append(yhat)
            year_list = county2['Year'].to_list()
            for i in range(1,7):
                year_list.append(year_list[-1] + 1)
            prediction_list = county2['KG (FULL DAY)'].to_list()
            prediction_list.extend(predictions)
            county_list = [county for i in range(len(prediction_list))]
            local = pd.DataFrame(np.column_stack([county_list, year_list,prediction_list]))
            dataframe_list.append(local)  
        except ValueError:
            continue
        except np.linalg.LinAlgError as err:
            continue
    else:
        continue
    
#%%
concatenated = pd.concat(dataframe_list, axis = 0)
#%%
concatenated.columns = ["County", "Year", "Enrollment"]
#%%
concatenated.to_csv("County_Predictions_Kg_Full_day.csv")
#%%


