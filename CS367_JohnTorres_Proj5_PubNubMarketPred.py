# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:04:00 2023

@author: John Torres
"""
#PubNub imports
from pubnub.callbacks import SubscribeCallback
from pubnub.enums import PNStatusCategory, PNOperationType
from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub

#Utility imports
import pandas as pd
import time
from csv import DictWriter
import matplotlib.pyplot as plt
import numpy as np

#SKlearn imports
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import RegressorChain

#PubNub config
pnconfig = PNConfiguration()
pnconfig.subscribe_key = 'sub-c-99084bc5-1844-4e1c-82ca-a01b18166ca8'
pnconfig.publish_key = 'pub-c-8429e172-5f05-45c0-a97f-2e3b451fe74f'
pnconfig.user_id = "my_custom_user_id"
pnconfig.connect_timeout = 5
pubnub = PubNub(pnconfig)


#marketkeys is a collection of keys that are present in each tweet, used in subscribe callback message
marketkeys = ['symbol', 'order_quantity', 'bid_price', 'trade_type', 'timestamp']

class MySubscribeCallback(SubscribeCallback):

    def status(self, pubnub, status):
        pass
        # PubNub setup
        if status.operation == PNOperationType.PNSubscribeOperation \
                or status.operation == PNOperationType.PNUnsubscribeOperation:
            if status.category == PNStatusCategory.PNConnectedCategory:
                pass
                # Is no error or issue whatsoever
            elif status.category == PNStatusCategory.PNReconnectedCategory:
                pass
                # If subscribe temporarily fails but reconnects. This means
                # there was an error but there is no longer any issue
            elif status.category == PNStatusCategory.PNDisconnectedCategory:
                pass
                # No error in unsubscribing from everything
            elif status.category == PNStatusCategory.PNUnexpectedDisconnectCategory:
                pass
                # This is an error, retry will be called automatically
            elif status.category == PNStatusCategory.PNAccessDeniedCategory:
                pass
                # This means that Access Manager does not allow this client to subscribe to this
                # channel and channel group configuration. This is another explicit error
            else:
                pass
                # This is usually an issue with the internet connection, this is an error, handle appropriately
                # retry will be called automatically
        elif status.operation == PNOperationType.PNSubscribeOperation:
            # Heartbeat operations can in fact have errors, so it is important to check first for an error.
            # For more information on how to configure heartbeat notifications through the status
            # PNObjectEventListener callback, consult <link to the PNCONFIGURATION heartbeart config>
            if status.is_error():
                pass
                # There was an error with the heartbeat operation, handle here
            else:
                pass
                # Heartbeat operation was successful
        else:
            pass
            # Encountered unknown status type
 
    def presence(self, pubnub, presence):
        pass  # handle incoming presence data
    def message(self, pubnub, message):
        res = dict()
        for key, val in message.message.items():
            if key in marketkeys:
                res[key] = val
        with open('p5output.csv', 'a', encoding="utf-8") as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=marketkeys)
            dictwriter_object.writerow(res)
            f_object.close()
        pass  # handle incoming message data
        
subscribeCallback = MySubscribeCallback()
pubnub.add_listener(subscribeCallback)
pubnub.subscribe().channels('pubnub-market-orders').execute()




## Increase time.sleep from (1) to higher number to increase amount of data gathered to .csv.
time.sleep(1)
pubnub.unsubscribe_all()


## Read in .csv.
pubnubmarketdf = pd.read_csv('p5output.csv', names=marketkeys)
print("\nShape of PubNub Market DF is", pubnubmarketdf.shape)


## Create new dataframe with all Google market orders and print shape.
googledf = pubnubmarketdf[pubnubmarketdf['symbol']=='Google']
print("\nShape of Google slice of PubNub Market DF is", googledf.shape)


## Create plot of Google bid prices.
fig, ax = plt.subplots(figsize=(16, 11))
ax.plot(googledf['bid_price'])
ax.set_xlabel('Timestamp')
ax.set_ylabel('Bid Price')
fig.autofmt_xdate()
plt.tight_layout()


## Slice out timestamp and bid_price columns and create learning dataframe.
selected_cols = ['timestamp', 'bid_price']
learndf = googledf[selected_cols].copy()
## Set index as timestamp
learndf.index = learndf['timestamp']
print("\nShape of Learn DF slice from Google slice of PubNub Market DF is", learndf.shape)


## Create y column and split learning dataframe into train and test sets.
learndf['y'] = learndf['bid_price'].shift(-1)
train = learndf[:-204]
test = learndf[-204:]


## Print shapes of training and test sets.
print("\nShape of training set is", train.shape)
print("\nShape of test set is", test.shape)


## Create baseline prediction column
test = test.copy()
test['baseline_pred'] = test['bid_price']
## Drop last row
test = test.drop(test.tail(1).index)


## Split training set into X & y train sets, reshape X_test
X_train = train['bid_price'].values.reshape(-1,1)
y_train = train['y'].values.reshape(-1,1)
X_test = test['bid_price'].values.reshape(-1,1)


## Create Decision Tree Regressor, fit with Training Models, make Prediction
dt_reg = DecisionTreeRegressor(random_state=42)
dt_reg.fit(X=X_train, y=y_train)
dt_pred = dt_reg.predict(X_test)


## Assign predictions to a new column in test
test['dt_pred'] = dt_pred


## MAPE calculation function
def mape(y_true, y_pred):
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)


## Calculate MAPE for Baseline and Decision Tree.
baseline_mape = mape(test['y'], test['baseline_pred'])
dt_mape = mape(test['y'], test['dt_pred'])


## Create Gradient Boosting Regressor, fit with Training Models, make Prediction.
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y=y_train.ravel())
gbr_pred = gbr.predict(X_test)
test['gbr_pred'] = gbr_pred


## Calculate MAPE for Gradient Boosting
gbr_mape = mape(test['bid_price'], test['gbr_pred'])


## Print MAPE values.
print(f'Baseline: {baseline_mape}%')
print(f'Decision Tree: {dt_mape}%')
print(f'Gradient Boosting Regressor: {gbr_mape}%')


## Create window dataframe function
def window_input(window_length: int, data: pd.DataFrame) -> pd.DataFrame:
    
    df = data.copy()
    
    i = 1
    while i < window_length:
        df[f'x_{i}'] = df['bid_price'].shift(-i)
        i = i + 1
        
    if i == window_length:
        df['y'] = df['bid_price'].shift(-i)
        
    # Drop rows where there is a NaN
    df = df.dropna(axis=0)
        
    return df


## Create Google window dataframe
googledf_window = window_input(5, googledf)
X = googledf_window[['bid_price', 'x_1', 'x_2', 'x_3', 'x_4']].values
y = googledf_window['y'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)


## Create Baseline
baseline_pred = []
## Append predictions to Baseline
for row in X_test:
    baseline_pred.append(np.mean(row))


## Create Decision Tree Regressor, fit with Training Models, make Prediction    
dt_reg_5 = DecisionTreeRegressor(random_state=42)
dt_reg_5.fit(X_train, y_train)
dt_reg_5_pred = dt_reg_5.predict(X_test)


## Create Gradient Boosting Regressor, fit with Training Models, make Prediction
gbr_5 = GradientBoostingRegressor(random_state=42)
gbr_5.fit(X_train, y_train.ravel())
gbr_5_pred = gbr_5.predict(X_test)


## Get MAPE values for Decision Tree Regressor, Gradient Boosting Regressor, and Baseline
baseline_mape = mape(y_test, baseline_pred)
dt_5_mape = mape(y_test, dt_reg_5_pred)
gbr_5_mape = mape(y_test, gbr_5_pred)


## Print MAPE values for window df.
print(f'Baseline MAPE: {baseline_mape}%')
print(f'Decision Tree MAPE: {dt_5_mape}%')
print(f'Gradient Boosting MAPE: {gbr_5_mape}%')


## Prepare Data for Sequence analysis
def window_input_output(input_length: int, output_length: int, data: pd.DataFrame) -> pd.DataFrame:
    
    df = data.copy()
    
    i = 1
    while i < input_length:
        df[f'x_{i}'] = df['bid_price'].shift(-i)
        i = i + 1
        
    j = 0
    while j < output_length:
        df[f'y_{j}'] = df['bid_price'].shift(-output_length-j)
        j = j + 1
        
    df = df.dropna(axis=0)
    
    return df

seq_df = window_input_output(26, 26, googledf)
X_cols = [col for col in seq_df.columns if col.startswith('x')]
X_cols.insert(0, 'bid_price')
y_cols = [col for col in seq_df.columns if col.startswith('y')]


## Create Training Sets from Sequence
X_train = seq_df[X_cols][:-2].values
y_train = seq_df[y_cols][:-2].values


## Create Test sets from Sequence
X_test = seq_df[X_cols][-2:].values
y_test = seq_df[y_cols][-2:].values


## Create Decision Tree Regressor Sequence, fit with Training Models, make Prediction
dt_seq = DecisionTreeRegressor(random_state=42)
dt_seq.fit(X_train, y_train)
dt_seq_preds = dt_seq.predict(X_test)


## Create Gradient Boosting Regressor Sequence, fit with Training Models, make Prediction
gbr_seq = GradientBoostingRegressor(random_state=42)
chained_gbr = RegressorChain(gbr_seq)
chained_gbr.fit(X_train, y_train)
gbr_seq_preds = chained_gbr.predict(X_test)


## Get MAPE values for Decision Tree Sequence, Gradient Boosting Sequence, and Baseline Sequence
mape_dt_seq = mape(dt_seq_preds.reshape(1, -1), y_test.reshape(1, -1))
mape_gbr_seq = mape(gbr_seq_preds.reshape(1, -1), y_test.reshape(1, -1))
mape_baseline = mape(X_test.reshape(1, -1), y_test.reshape(1, -1))


## Plot input/Actual/Baseline/Decision Tree/Gradient Boosting
fig, ax = plt.subplots(figsize=(16, 11))

ax.plot(np.arange(0, 26, 1), X_test[1], 'b-', label='input')
ax.plot(np.arange(26, 52, 1), y_test[1], marker='.', color='blue', label='Actual')
ax.plot(np.arange(26, 52, 1), X_test[1], marker='o', color='red', label='Baseline')
ax.plot(np.arange(26, 52, 1), dt_seq_preds[1], marker='^', color='green', label='Decision Tree')
ax.plot(np.arange(26, 52, 1), gbr_seq_preds[1], marker='P', color='black', label='Gradient Boosting')

ax.set_xlabel('Timesteps')
ax.set_ylabel('Bid Prices')

plt.xticks(np.arange(1, 104, 52), np.arange(2000, 2002, 1))
plt.legend(loc=2)

fig.autofmt_xdate()
plt.tight_layout()