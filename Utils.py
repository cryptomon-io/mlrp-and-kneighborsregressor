import numpy as np
import sys
import time;
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib.pyplot as mpl
import matplotlib.collections as collections

import sys
sys.path.append("../API_Client/") 

from ApiClient import ApiClient

#Fetch data from API
#
#client_id:       API Client ID
#client_secret:   API Client ID
#market:          Candle market
#currency_pair:   Candle currency pair
#period:          Candle period
def fetch_data_from_api(client_id, client_secret, market, currency_pair, period):
    
    c = ApiClient(client_id, client_secret)
    c.request_access()

    params = {
        "market" : market,
        "currencyPair" : currency_pair,
        "period" : period
    }
    rows = c.get("https://api.cryptomon.io/api/v1/candles", params)
    
    row_list=[]
    print("\n")
    print("Total rows from API:", len(rows))
    for row in rows:
        timestamp = row["timestamp"] / 1000
        open = row["open"]
        close = row["close"]
        high = row["high"]
        low = row["low"]
        volume = row["volume"]
        # print(timestamp, open, high, low, close, volume, close)
        row_list.append([open, high, low, close, volume, close, timestamp])

    columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Timestamp']
    df = pd.DataFrame(row_list, columns=columns)

    return df

#Given dataframe from ParseData
#plot it to the screen
#
#df:        Dataframe returned from
#p:         The position of the predicted data points
def plot_data(df, p = None):
    if(p is None):
        p = np.array([])
    #Timestamp data
    ts = df.Timestamp.values
    #Number of x tick marks
    nTicks= 10
    #Left most x value
    s = np.min(ts)
    #Right most x value
    e = np.max(ts)
    #Total range of x values
    r = e - s
    #Add some buffer on both sides
    s -= r / 5
    e += r / 5
    #These will be the tick locations on the x axis
    tickMarks = np.arange(s, e, (e - s) / nTicks)
    #Convert timestamps to strings
    strTs = [datetime.fromtimestamp(i).strftime('%m-%d-%yT%H:%M:%S') for i in tickMarks]
    mpl.figure()
    #Plots of the high and low values for the day
    mpl.plot(ts, df.High.values, color = '#727272', linewidth = 1.618, label = 'Actual')
    #Predicted data was also provided
    if(len(p) > 0):
        mpl.plot(ts[p], df.High.values[p], color = '#7294AA', linewidth = 1.618, label = 'Predicted')
    #Set the tick marks
    mpl.xticks(tickMarks, strTs, rotation='vertical')
    #Set y-axis label
    mpl.ylabel('Token High Value (USD)')
    #Add the label in the upper left
    mpl.legend(loc = 'upper left')
    mpl.show()
    

#Gives a list of timestamps from the start date to the end date
#
#startDate:     The start date as a string xxxx-xx-xx xx:xx:xx
#endDate:       The end date as a string year-month-day hour:minutes:sec
#period:		'daily', 'weekly', or 'monthly'
#weekends:      True if weekends should be included; false otherwise
#return:        A numpy array of timestamps
def date_range(startDate, endDate, period, weekends = True):
    #The start and end date
    sd = datetime.strptime(startDate, '%Y-%m-%dT%H:%M:%S')
    ed = datetime.strptime(endDate, '%Y-%m-%dT%H:%M:%S')

    print ("period: {}".format(period))
    print ("sd: {}".format(sd))
    print ("ed: {}".format(ed))

    #Invalid start and end dates
    if(sd > ed):
        raise ValueError("The start date cannot be later than the end date.")
    #One time period is a hour
    if(period == 'ONE_HOUR'):
        prd = timedelta(hours=1)
    #One time period is a day
    elif(period == 'ONE_DAY'):
        prd = timedelta(hours=24)
    #One prediction per week
    elif(period == 'WEEK'):
        prd = timedelta(hours=168)
    #one prediction every 30 days ("month")
    else:
        prd = timedelta(hours=5040)
    #The final list of timestamp data
    dates = []
    cd = sd
    while(cd <= ed):
        #If weekdays are included or it's a weekday append the current ts
        if(weekends or (cd.date().weekday() != 5 and cd.date().weekday() != 6)):
            dates.append(cd.timestamp())
        #Onto the next period
        cd = cd + prd
    return np.array(dates)


#Given a date, returns the previous day
#
#startDate:     The start date as a datetime object
#weekends:      True if weekends should counted; false otherwise
def date_prev_day(startDate, weekends = True):
    #One day
    day = timedelta(hours=24)
    cd = datetime.fromtimestamp(startDate)
    while(True):
        cd = cd - day
        if(weekends or (cd.date().weekday() != 5 and cd.date().weekday() != 6)):
            return cd.timestamp()
    #Should never happen
    return None