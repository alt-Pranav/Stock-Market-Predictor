import requests, io

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
plt.set_loglevel('WARNING') 

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
# from pmdarima import auto_arima

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

import pickle5 as pickle


st.title('Stock Prediction App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'TSLA', 'IBM')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# n_years = st.slider('Years of prediction:', 1, 10)
# period = 1 * 365

# @suppress_warning
@st.cache
def load_data(symbol):
    api_key = 'LMONBJ0BJUABS722'

    # get daily data for symbol
    url_daily = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol='+symbol+'&apikey='+api_key+'&datatype=csv'

    # get weekly data for GOOG
    # url_weekly = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol='+symbol+'&apikey='+api_key+'&datatype=csv'

    r_daily = requests.get(url_daily)
    # r_weekly = requests.get(url_weekly)

    dailyData = r_daily.content
    # weeklyData = r_weekly.content

    df_daily = pd.read_csv(io.StringIO(dailyData.decode('utf-8')))
    #df_weekly = pd.read_csv(io.StringIO(weeklyData.decode('utf-8')))

    return df_daily

data_load_state = st.text('Loading data...')
df_daily = load_data(selected_stock)
data_load_state.text('Loading data... done!')


st.subheader('Raw data')
st.write(df_daily.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df_daily['timestamp'], y=df_daily['open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=df_daily['timestamp'], y=df_daily['close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


#The columns Open and Close represent the starting and final price at which the stock is traded on a particular day.
#High and Low represent the maximum and minimum of the share for the day.
#Date i.e. timestamp is already in %Y-%M-%D format
# print(df_weekly.shape)

# df_weekly.tail()

# def saveAsCSV(df_daily, df_weekly):
#   df_daily.to_csv('daily_'+symbol+'.csv', index=False)
#   df_weekly.to_csv('weekly_'+symbol+'.csv', index=False)

# df_daily

df_daily['Date'] = pd.to_datetime(df_daily['timestamp'])
df_daily2 = df_daily.set_axis(df_daily['Date'])
#df_daily.drop(columns=['open', 'high', 'low', 'volume'], inplace=True)


df_daily2 = df_daily2.sort_index(ascending=True)

print("========================================")
print(df_daily2)


close_data = df_daily2['close'].values
close_data = close_data.reshape((-1,1))

print(close_data[:5])

split_percent = 0.80
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df_daily2['timestamp'][:split]
date_test = df_daily2['timestamp'][split:]


print(close_train[0:5])

look_back = 15

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

data_load_state = st.text('Training/Loading Model...')

# model = Sequential()
# model.add(
#     LSTM(10,
#         activation='relu',
#         input_shape=(look_back,1))
# )
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

# num_epochs = 25
# model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

filename = f'{selected_stock}_2.sav'


###### dump model
# pickle.dump(model, open(filename, 'wb'))


####### load model
model = pickle.load(open(filename, 'rb'))

data_load_state.text('Model Loaded/Trained!')

close_data = close_data.reshape((-1))

def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = df_daily2['Date'].values[-1]
    print("================last date ============================")
    print(last_date)
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates


num_prediction = st.slider('Days of prediction:', 1, 15)

#num_prediction = 15
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)

print(forecast)

forecasted_dates = pd.to_datetime(forecast_dates)
print(forecasted_dates)

# plt.figure(figsize=(20,6))
# plt.plot(forecast_dates, forecast)

# fig, ax = plt.subplots(figsize=(30,6))


fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, name="forecast"))
#fig.add_trace(go.Scatter(x=valid.index, y=valid['close'], name="validation_close"))
#fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name="predicted_close"))
fig.layout.update(title_text='Future Forecast', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# valid = close_test


# valid_withTSG = close_test


# valid_withTSG['Predictions'].append(pd.Series(forecast))

# ax.plot(close_train['close'])
# ax.plot(close_valid[['close','Predictions']])
# ax.plot(forecast)

# plt.legend(["train_close", "valid_close", "valid_pred", "next week"])

# ax.plot(close_train['close'])
# ax.plot(valid_withTSG[['close','Predictions']])

# print(valid_withTSG)



