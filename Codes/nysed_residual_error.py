# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:59:32 2020

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
# load data
data2 = read_csv("Conty_level_enrollment_Data.csv")
data2['Year'] = data2['SCHOOL YEAR'].apply(lambda x: x[0:4])
#%%

#%%
dat3 = pd.read_excel("20200929_staff_turnover.xlsx")
#%%
#%%
dat3= dat3[['COUNTY_NAME', 'REGION_NAME']].drop_duplicates(subset = ['COUNTY_NAME', 'REGION_NAME'], keep = "first")
#%%
merge_data = pd.merge(data2, dat3, left_on = "COUNTY", right_on="COUNTY_NAME")
#%%
data_final = merge_data.groupby(['Year','REGION_NAME'])['PK (FULL DAY)'].sum()
#%%

data_final = data_final.reset_index()
# #%%
main_list = []

#%%
for county in data_final.REGION_NAME.unique():
    local_list = []
    albany = data_final[data_final.REGION_NAME == county]
    series = pd.Series(albany['PK (FULL DAY)'].to_list(), index=albany['Year'].to_list())# create lagged dataset
    values = DataFrame(series.values)
    values.dropna(inplace = True)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t', 't+1']
    dataframe.dropna(inplace = True)
    X = dataframe.values
    dict1= {}
    for size in [0.50,0.55, 0.60,0.66,0.70,0.75,0.80]:
        train_size = int(len(X) * size)
        train, test = X[1:train_size], X[train_size:]
        train_X, train_y = train[:,0], train[:,1]
        test_X, test_y = test[:,0], test[:,1]
        # persistence model on training set
        train_pred = [x for x in train_X]
        # calculate residuals
        train_resid = [train_y[i]-train_pred[i] for i in range(len(train_pred))]
        # model the training set residuals
        model = AR(train_resid)
        model_fit = model.fit()
        window = model_fit.k_ar
        coef = model_fit.params
        # walk forward over time steps in test
        history = train_resid[len(train_resid)-window:]
        history = [history[i] for i in range(len(history))]
        predictions = list()
        for t in range(len(test_y)):
        	# persistence
        	yhat = test_X[t]
        	error = test_y[t] - yhat
        	# predict error
        	length = len(history)
        	lag = [history[i] for i in range(length-window,length)]
        	pred_error = coef[0]
        	for d in range(window):
        		pred_error += coef[d+1] * lag[window-d-1]
        	# correct the prediction
        	yhat = yhat + pred_error
        	predictions.append(yhat)
        	history.append(error)
        	print('predicted=%f, expected=%f' % (yhat, test_y[t]))
        # error
        rmse = sqrt(mean_squared_error(test_y, predictions))
        print("Test Size:", size)
        print('Test RMSE: %.3f' % rmse)
        dict1[size] = rmse
    final_size = min(dict1, key=dict1.get)
    local_list.append(county)
    local_list.append(final_size)
    local_list.append(dict1[final_size])
    main_list.append(local_list)
#%%
dataframe1 = pd.DataFrame(main_list, columns = ["County", "Size", "RMSE"])   
    #%%
dataframe_list = []
#%%
for index,row in data2.iterrows():
    local_list =[]
    county = getattr(row, "COUNTY")
    print(county)
    #size = getattr(row, "Size")
    albany = data2[data2.COUNTY == county]
    series = pd.Series(albany['PK (FULL DAY)'].to_list(), index=albany['Year'].to_list())# create lagged dataset
    values = DataFrame(series.values)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t', 't+1']
    dataframe.dropna(inplace = True)
    X = dataframe.values
    train_size = int(len(X) * 0.90)
    train, test = X[1:train_size], X[train_size:]
    train_X, train_y = train[:,0], train[:,1]
    test_X, test_y = test[:,0], test[:,1]
    # persistence model on training set
    train_pred = [x for x in train_X]
    # calculate residuals
    train_resid = [train_y[i]-train_pred[i] for i in range(len(train_pred))]
    # model the training set residuals
    model = AR(train_resid)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    # walk forward over time steps in test
    history = train_resid[len(train_resid)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test_y)):
     	# persistence
     	yhat = test_X[t]
     	error = test_y[t] - yhat
     	# predict error
     	length = len(history)
     	lag = [history[i] for i in range(length-window,length)]
     	pred_error = coef[0]
     	for d in range(window):
    		pred_error += coef[d+1] * lag[window-d-1]
     	# correct the prediction
     	yhat = yhat + pred_error
     	predictions.append(yhat)
     	history.append(error)
     	print('predicted=%f, expected=%f' % (yhat, test_y[t]))
    # error
    rmse = sqrt(mean_squared_error(test_y, predictions))
    print("Test Size:", size)
    print('Test RMSE: %.3f' % rmse)
    for t in range(1,7):
        yhat = test_y[-1]
        error = predictions[-1] - yhat
        # predict error
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        pred_error = coef[0]
        for d in range(window):
     	    pred_error += coef[d+1] * lag[window-d-1]
     	# correct the prediction
        yhat = yhat + pred_error
        test_y = np.append(test_y, predictions[-1])
        predictions.append(yhat)
        history.append(error)
        print('predicted=%f' % (yhat))
    albany.Year = albany.Year.astype(int)
    year_list = albany['Year'].to_list()
    for i in range(1,7):
        year_list.append(year_list[-1] + 1)
    prediction_list = albany['PK (FULL DAY)'].to_list()
    prediction_list.extend(predictions[-6:])
    county_list = [county for i in range(len(prediction_list))]
    local = pd.DataFrame(np.column_stack([county_list, year_list,prediction_list]))
    dataframe_list.append(local)  
    except ValueError:
        continue
#%%
    
finalframe = pd.concat(dataframe_list) 
#%%
finalframe.columns = ["Region", "Year", "Enrollment"]
#%%
finalframe.to_csv("Region_Enrollments_final.csv", index = False)
#%%
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()
#%%
# albany = data2[data2.COUNTY == "NEW YORK"]
# series = pd.Series(albany['KG (FULL DAY)'].to_list(), index=albany['Year'].to_list())# create lagged dataset
# values = DataFrame(series.values)
# dataframe = concat([values.shift(1), values], axis=1)
# dataframe.columns = ['t', 't+1']
# dataframe.dropna(inplace = True)
# X = dataframe.values
# dict1= {}
# for size in [0.50,0.55, 0.60,0.66,0.70,0.75,0.80]:
#     train_size = int(len(X) * size)
#     train, test = X[1:train_size], X[train_size:]
#     train_X, train_y = train[:,0], train[:,1]
#     test_X, test_y = test[:,0], test[:,1]
#     # persistence model on training set
#     train_pred = [x for x in train_X]
#     # calculate residuals
#     train_resid = [train_y[i]-train_pred[i] for i in range(len(train_pred))]
#     # model the training set residuals
#     model = AR(train_resid)
#     model_fit = model.fit()
#     window = model_fit.k_ar
#     coef = model_fit.params
#     # walk forward over time steps in test
#     history = train_resid[len(train_resid)-window:]
#     history = [history[i] for i in range(len(history))]
#     predictions = list()
#     for t in range(len(test_y)):
#     	# persistence
#     	yhat = test_X[t]
#     	error = test_y[t] - yhat
#     	# predict error
#     	length = len(history)
#     	lag = [history[i] for i in range(length-window,length)]
#     	pred_error = coef[0]
#     	for d in range(window):
#     		pred_error += coef[d+1] * lag[window-d-1]
#     	# correct the prediction
#     	yhat = yhat + pred_error
#     	predictions.append(yhat)
#     	history.append(error)
#     	print('predicted=%f, expected=%f' % (yhat, test_y[t]))
#     # error
#     rmse = sqrt(mean_squared_error(test_y, predictions))
#     print("Test Size:", size)
#     print('Test RMSE: %.3f' % rmse)
#     dict1[size] = rmse
# #%%
#%%
dataframe_list = []
#%%
main_cot_list = []
main_numberlist = []
main_year_list = []


#%%
from math import log
from math import exp
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)

for county in data2.COUNTY.unique():
    county1 = data2[data2.COUNTY == county]
    county1.fillna(0, inplace = True)
    county1.set_index("Year", inplace = True)
    value = county1.loc['2019','PK (FULL DAY)']
    try:
        print(county)
        county1.reset_index(inplace = True)
        county1.Year = county1.Year.astype(int)
        county2 = county1[county1['PK (FULL DAY)'] > 0]
        series = pd.Series(county2['PK (FULL DAY)'].to_list(), index=county2.Year.to_list())
        X = series.values
        X = X.astype('float32')
        # train_size = int(len(X) * 0.80)
        # train, test = X[0:train_size], X[train_size:]
        # walk-forward validation
        history = [x for x in X]
        predictions = list()
        for i in range(1,7):
            transformed, lam = boxcox(history)
            if lam <-5:
                transformed, lam = history , 1
            model = ARIMA(transformed, order = (4,0,0))
            model_fit = model.fit(disp=0)
            yhat = model_fit.forecast()[0]
            yhat = boxcox_inverse(yhat, lam)
            yhat = np.round(yhat,0)
            predictions.append(yhat)
            history.append(yhat)
        year_list = county2['Year'].to_list()
        for i in range(1,7):
            year_list.append(year_list[-1] + 1)
        print(year_list)
        prediction_list = county2['PK (FULL DAY)'].to_list()
        prediction_list.extend(predictions)
        print(prediction_list)
        county_list = [county for i in range(len(prediction_list))]
        main_cot_list.extend(county_list)
        main_year_list.extend(year_list)
        main_numberlist.extend(prediction_list)
    except ValueError:
        continue
    except np.linalg.LinAlgError as err:
        continue
#%%
local = pd.DataFrame(np.column_stack([main_cot_list, main_year_list,main_numberlist]))
  
    
#%%
concatenated = pd.concat(dataframe_list, axis = 0)
#%%
local.columns = ["County", "Year", "Enrollment"]
#%%
local.to_csv("County_Predictions_PK_Full_day_Final.csv")