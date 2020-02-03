# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn import tree
from sklearn import metrics 
from sklearn.preprocessing import quantile_transform
from sklearn.preprocessing import QuantileTransformer
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler


# load solararray complete data using pandas
features=pd.read_csv('../input/df_solararray_complete.csv')
#features.head(3)


#X=features.drop(['Unnamed: 0', 'Location', 'Year', 'Month','Day', 'Dew_Point','Wind_Speed','Precipitation','Pressure'], axis=1)
X=pd.DataFrame(features[['Solar_Elevation', 'Hour', 'Month', 'Electricity_KW_HR']])
#names=list(X.columns)
#X.head(3)

# labels are the values we want to predict
y=np.array(X['Electricity_KW_HR'])

# Remove the labels from the features
# axis 1 refers to the columns
X=X.drop('Electricity_KW_HR',axis=1)

# saving feature names for later use
#feature_list=list(X.columns)

#convert to numpy array
X=np.array(X)


# load scenario (test file) data using pandas and then sort the values by hour to keep the date and the hours the same
features_snr=pd.read_csv('../input/scenario.csv')

# Now let's fill missing values for those variables which we are interested on and others which we are dropping later
features_snr['Visibility'].fillna(value=features_snr['Visibility'].mean(), inplace=True)
features_snr['Temperature'].fillna(value=features_snr['Temperature'].mean(), inplace=True)
features_snr['Humidity_Fraction'].fillna(value=features_snr['Humidity_Fraction'].mean(), inplace=True)
features_snr['Cloud_Cover_Fraction'].fillna(value=features_snr['Cloud_Cover_Fraction'].mean(), inplace=True)

#Now let's take out only those 6 days
a=features_snr.query('Month == 3 & Day == 15').sort_values(by=['Hour'])
b=features_snr.query('Month == 6 & Day == 26').sort_values(by=['Hour'])
c=features_snr.query('Month == 7 & Day == 3').sort_values(by=['Hour'])
d=features_snr.query('Month == 10 & Day == 13').sort_values(by=['Hour'])
e=features_snr.query('Month == 11 & Day == 19').sort_values(by=['Hour'])
f=features_snr.query('Month == 12 & Day == 25').sort_values(by=['Hour'])
test_dates_df = pd.concat([a, b,c,d,e,f]) #This line is for all dates
#test_dates_df = pd.concat([a]) #, b,c,d,e,f]) # this line is for only one date
test_dates_df.head()

#test_dates_df=test_dates_df.drop(['Unnamed: 0', 'City', 'Month', 'Day', 'Year', 'Day_of_week', 'HolidayName', 'School_Day', 'Weekdays', 'Dew_Point', 'Precipitation', 'Pressure', 'Wind_Speed', 'Cloud_Cover_Fraction', 'Humidity_Fraction', 'Temperature', 'Visibility'], axis=1)
test=pd.DataFrame(test_dates_df[['Solar_Elevation','Hour', 'Month']])
test.head()

############# minMaxScaler #################
def scaler():
    pt1 = MinMaxScaler()
    pt2 = MinMaxScaler()
    X_t=pt1.fit_transform(X)
    y_t=pt2.fit_transform(y.reshape(-1,1))
    return pt1, pt2, X_t, y_t

############# let's try Yeo-Johnson transformation #################
# normalize using Yeo-Johnson:
def yeoJohnson():
    pt1 = PowerTransformer()
    pt2 = PowerTransformer()
    X_t=pt1.fit_transform(X) 
    y_t=pt2.fit_transform(y.reshape(-1,1))
    return pt1, pt2, X_t, y_t

############# let's try Quantile transformation #################
# normalize using quantile
#features=quantile_transform(features, n_quantiles=10, random_state=0, copy=True)
def quantile():
    pt1 = QuantileTransformer(n_quantiles=10, random_state=0)
    pt2 = QuantileTransformer(n_quantiles=10, random_state=0)
    X_t=pt1.fit_transform(X) 
    y_t=pt2.fit_transform(y.reshape(-1,1))
    return pt1, pt2, X_t, y_t




# Now let's predict the Electricity_KW_Hr on these 6 test dates using linear model
############################ LINEAR MODEL ##################
def predLinearMinMax():
    lm = linear_model.LinearRegression()
    lm.fit(X_t, y_t) # Train the model using the training sets
    r2=lm.score(X_t, y_t) # Make predictions using the testing set
    pt3 = MinMaxScaler()
    test_t=pt3.fit_transform(test)
    pred_Elec_t=lm.predict(test_t)
    pred_Elec=pt2.inverse_transform(pred_Elec_t.reshape(-1,1))
    return pred_Elec, r2

def predLinearYJ():
    lm = linear_model.LinearRegression()
    lm.fit(X_t, y_t) # Train the model using the training sets
    r2=lm.score(X_t, y_t) # Make predictions using the testing set
    pt3 = PowerTransformer()
    test_t=pt3.fit_transform(test)
    pred_Elec_t=lm.predict(test_t)
    pred_Elec=pt2.inverse_transform(pred_Elec_t.reshape(-1,1))
    return pred_Elec, r2

def predLinearQT():
    lm = linear_model.LinearRegression()
    lm.fit(X_t, y_t) # Train the model using the training sets
    r2=lm.score(X_t, y_t) # Make predictions using the testing set
    pt3 = QuantileTransformer(n_quantiles=10, random_state=0)
    test_t=pt3.fit_transform(test)
    pred_Elec_t=lm.predict(test_t)
    pred_Elec=pt2.inverse_transform(pred_Elec_t.reshape(-1,1))
    return pred_Elec, r2

############################ RANDOM FOREST MODEL ##################
def predRFMinMax():
    rf = RandomForestRegressor(n_estimators = 1400, random_state = 42)
    rf.fit(X_t, y_t)  # Train the model on training data
    r2=rf.score(X_t, y_t.ravel()) # Make predictions using the testing set
    pt3 = MinMaxScaler()
    test_t=pt3.fit_transform(test)
    pred_Elec_t=rf.predict(test_t)
    pred_Elec=pt2.inverse_transform(pred_Elec_t.reshape(-1,1))
    return pred_Elec, r2

def predRFYJ():
    rf = RandomForestRegressor(n_estimators = 1400, random_state = 42)
    rf.fit(X_t, y_t)  # Train the model on training data
    r2=rf.score(X_t, y_t) # Make predictions using the testing set
    pt3 = PowerTransformer()
    test_t=pt3.fit_transform(test)
    pred_Elec_t=rf.predict(test_t)
    pred_Elec=pt2.inverse_transform(pred_Elec_t.reshape(-1,1))
    return pred_Elec, r2

def predRFQT():
    rf = RandomForestRegressor(n_estimators = 1400, random_state = 42)
    rf.fit(X_t, y_t)  # Train the model on training data
    r2=rf.score(X_t, y_t) # Make predictions using the testing set
    pt3 = QuantileTransformer(n_quantiles=10, random_state=0)
    test_t=pt3.fit_transform(test)
    pred_Elec_t=rf.predict(test_t)
    pred_Elec=pt2.inverse_transform(pred_Elec_t.reshape(-1,1))
    return pred_Elec, r2

############################ XGboost MODEL ##################
def predXgbMinMax():
    xg = xgb.XGBRegressor()
    xg.fit(X_t, y_t)
    r2=xg.score(X_t, y_t)
    pt3 = MinMaxScaler()
    test_t=pt3.fit_transform(test)
    pred_Elec_t=xg.predict(test_t)
    pred_Elec=pt2.inverse_transform(pred_Elec_t.reshape(-1,1)) 
    return pred_Elec, r2

def predXgbYJ():
    xg = xgb.XGBRegressor()
    xg.fit(X_t, y_t)
    r2=xg.score(X_t, y_t)
    pt3 = PowerTransformer()
    test_t=pt3.fit_transform(test)
    pred_Elec_t=xg.predict(test_t)
    pred_Elec=pt2.inverse_transform(pred_Elec_t.reshape(-1,1)) 
    return pred_Elec, r2
    
def predXgbQT():
    xg = xgb.XGBRegressor()
    xg.fit(X_t, y_t)
    r2=xg.score(X_t, y_t)
    pt3 = QuantileTransformer(n_quantiles=10, random_state=0)
    test_t=pt3.fit_transform(test)
    pred_Elec_t=xg.predict(test_t)
    pred_Elec=pt2.inverse_transform(pred_Elec_t.reshape(-1,1))
    return pred_Elec, r2

    
### MinMaxScaler and Linear
#pt1, pt2, X_t, y_t = scaler()
#pred_Elec, r2 =predLinearMinMax()

### Yeo-Johnson and Linear
#pt1, pt2, X_t, y_t = yeoJohnson()
#pred_Elec, r2=predLinearYJ()

### Quantile and Linear
#pt1, pt2, X_t, y_t = yeoJohnson()
#pred_Elec, r2=predLinearQT()

### MinMaxScaler and RF
#pt1, pt2, X_t, y_t = scaler()
#pred_Elec,r2=predRFMinMax()

#predMMRF=pred_Elec

### Yeo-Johnson and RF
#pt1, pt2, X_t, y_t = yeoJohnson()
#pred_Elec, r2=predRFYJ()

### Quantile and RF
#pt1, pt2, X_t, y_t = quantile()
#pred_Elec, r2=predRFQT()

### MinMaxScaler and XGb
pt1, pt2, X_t, y_t = scaler()
pred_Elec, r2 =predXgbMinMax()
#predMMXGB=pred_Elec


### Yeo-Johnson and XGb
#pt1, pt2, X_t, y_t = yeoJohnson()
#pred_Elec, r2=predXgbYJ()

### Quantile and Xgb
#pt1, pt2, X_t, y_t = quantile()
#pred_Elec,r2=predXgbQT()





# Now let's find the RMSE of these predicted values agest the actual one imputed from the training dataset
g=features.query('Month == 3 & Day == 15').sort_values(by=['Hour']).groupby(['Month','Day','Hour'], as_index=False).agg(np.mean)
h=features.query('Month == 6 & Day == 26').sort_values(by=['Hour']).groupby(['Month','Day','Hour'], as_index=False).agg(np.mean)
i=features.query('Month == 7 & Day == 3').sort_values(by=['Hour']).groupby(['Month','Day','Hour'], as_index=False).agg(np.mean)
j=features.query('Month == 10 & Day == 13').sort_values(by=['Hour']).groupby(['Month','Day','Hour'], as_index=False).agg(np.mean)
k=features.query('Month == 11 & Day == 19').sort_values(by=['Hour']).groupby(['Month','Day','Hour'], as_index=False).agg(np.mean)
l=features.query('Month == 12 & Day == 25').sort_values(by=['Hour']).groupby(['Month','Day','Hour'], as_index=False).agg(np.mean)

# take out the Electricity_KW_HR column only since that is what we are inerested in.
g=g['Electricity_KW_HR']
h=h['Electricity_KW_HR']
i=i['Electricity_KW_HR']
j=j['Electricity_KW_HR']
k=k['Electricity_KW_HR']
l=l['Electricity_KW_HR']

# concatenate these values
actual_Elec=pd.concat([g,h, i, j, k, l]) # this line concatenates all six dates
#actual_Elec=pd.concat([h]) #,h, i, j, k, l]) # this line only does one date at a time


# find the MSE and RMSE
MSE=mean_squared_error(actual_Elec, pred_Elec, sample_weight=None, multioutput='uniform_average')
RMSE=np.sqrt(mean_squared_error(actual_Elec, pred_Elec, sample_weight=None, multioutput='uniform_average'))
MAE = mean_absolute_error(actual_Elec, pred_Elec, sample_weight=None, multioutput='uniform_average')

## This will be used only with the final picked model; after that this values will be transfered to the excel table that Sierra put up.
#SixDayPred=pd.concat([pred_Elec])

# Plot the output for the linear regression
fig, ax = plt.subplots()
ax=plt.plot(np.array(actual_Elec), alpha=1, color='red')
ax=plt.plot(pred_Elec, alpha=1, color='b')
plt.grid()
#plt.xticks(np.arange(0, 25, 1)) 
plt.xlabel('Hours')
plt.ylabel('Electricity (KW/h)')
plt.title('Actual vs. Predicted Electricity (KW/h)\n XGb Model with Yeo-Johnson Transformation\nFor March 15th Date')
plt.suptitle('Random Forest with Min-Max transformation with MSE = {}, RMSE = {}, MAE={}, r2 = {}'.format(MSE, RMSE, MAE, r2), fontsize=16)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20,15]
green_patch = mpatches.Patch(color='b', label='Predicted Electricity (KW/h)')
orange_patch = mpatches.Patch(color='red', label='Actual Electricity (KW/h)')
plt.legend(handles=[green_patch, orange_patch], loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()



###### let's take out the actual data for KWh for 2012 for 3/15, 6/26, 7/3, 10/13, 11/19, 12/25 ######
#d=[3,6,7,10,11,12]
#e=[15,26,3,13,19,25]
#new=pd.DataFrame(features[['Year', 'Month','Day','Hour', 'Electricity_KW_HR']])
#for i in range(6):
#    print(new.query('Year==2012 & Month == {} & Day == {}'.format(d[i], e[i])).sort_values(by=['Hour']))


#pt1, pt2, X_t, y_t = quantile()
for i in range(144):
    print(pred_Elec[i][0])
#import sys
#np.savetxt(sys.stdout, pred_Elec)

# Plot the output for the linear regression
fig, ax = plt.subplots()
ax=plt.plot(np.array(actual_Elec), alpha=1, color='red', linewidth=5.0)
ax=plt.plot(predMMRF, alpha=0.7, color='b', ls='--', linewidth=5.0)
ax=plt.plot(predMMXGB, alpha=0.7, color='g', ls='--', linewidth=5.0)
plt.grid()
#plt.xticks(np.arange(0, 25, 1)) 
plt.xlabel('Hours')
plt.ylabel('Electricity (KW/h)')
plt.title('Actual vs. Predicted Electricity (KW/h)\n RF and XGb Model with MinMax Transformation\nFor March 15th Date')
plt.suptitle('Random Forest with Min-Max transformation with MSE = {}, RMSE = {}, MAE={}, r2 = {}'.format(MSE, RMSE, MAE, r2), fontsize=16)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20,15]
blue_patch = mpatches.Patch(color='b', label='Predicted RF with min max scaler Electricity (KW/h)')
red_patch = mpatches.Patch(color='red', label='Actual Electricity (KW/h)')
green_patch = mpatches.Patch(color='green', label='Actual XGB wih min max scaler Electricity (KW/h)')
plt.legend(handles=[blue_patch, red_patch,green_patch], loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()