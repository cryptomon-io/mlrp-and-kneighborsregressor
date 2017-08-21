# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from Predictor import Predictor
from sklearn.neighbors import KNeighborsRegressor
#Used to get command line arguments
import sys
import os
#Used to check validity of date
from datetime import datetime, timedelta

from TFMLP import MLPR

import Utils
from Utils import fetch_data_from_api, plot_data

#Display usage information
def print_usage():
    print('Usage:\n')
    print('\tpython main.py <source type> <market> <currency pair> <period> <start time> <end time>')

    print('\t')
    print('\tsource type: K_NEIGHBORS_V1')
    print('\tsource type: MLPR_V1')
    print('\t')
    print('\tmarket: BITSTAMP')
    print('\t')
    print('\tcurrency pair: BTC_USD')
    print('\t')
    print('\tperiod: ONE_HOUR')
    print('\tperiod: ONE_DAY')
    print('\tperiod: WEEK')
    print('\t')
    print('\tstart time: NOW | 2017-02-09T08:00:00')
    print('\t')
    print('\tend time (optional): 2017-03-14T16:00:00')

#Main program
def main(args):
    if(len(args) != 5 and len(args) != 6):
        print("Invalid parameters count")
        print_usage()
        return

    print("-------------------------------------------------------------------")
    print ("Start time = %s" % datetime.now())
    print("\n");

    #API Client ID
    try:
        api_client_id = os.environ["API_CLIENT_ID"]
    except KeyError:
        print("Env. variable API_CLIENT_ID is not defined or is invalid!")
        print_usage()
        return

    #API Client secret
    try:
        api_client_secret = os.environ["API_CLIENT_SECRET"]
    except KeyError:
        print("Env. variable API_CLIENT_SECRET is not defined or is invalid!")
        print_usage()
        return

    #Source type
    pred_source = args[0].upper()

    #Market type
    market = args[1].upper()

    #Currency pair
    currency_pair = args[2].upper()

    #Period
    period = args[3].upper()

    # Start and end time
    if(args[4] == "NOW"):
        if(period == "ONE_HOUR"):
            startTime = datetime.today()
            # startTime = startTime - timedelta(hours=24 + 3)
            startTime = startTime - timedelta(hours=0)
            startTime = startTime.replace(minute=0, second=0)
            endTime = startTime + timedelta(hours=24*3) # 3 dni
            endTime = endTime.replace(minute=0, second=0)
        elif(period == "ONE_DAY"):
            startTime = datetime.today()
            # startTime = startTime - timedelta(days=14) # 16 trochu rozdielne
            startTime = startTime - timedelta(days=0)
            startTime = startTime.replace(hour=0, minute=0, second=0)
            endTime = startTime + timedelta(days=30*5) # 5mesiace
            endTime = endTime.replace(hour=0, minute=0, second=0)
        elif(period == "WEEK"):
            startTime = datetime.today()
            # startTime = startTime - timedelta(weeks=14)
            startTime = startTime - timedelta(weeks=0)
            while startTime.weekday() != 0: #0 for monday
                startTime -= timedelta(days=1)

            startTime = startTime.replace(hour=0, minute=0, second=0)
            endTime = startTime + timedelta(weeks=150)
            while endTime.weekday() != 0: #0 for monday
                endTime -= timedelta(days=1)
            endTime = endTime.replace(hour=0, minute=0, second=0)

            startTime += timedelta(days=1)
            endTime += timedelta(days=1)

        start = startTime.strftime("%Y-%m-%dT%H:%M:%S")
        end = endTime.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        #Test validity of start date string
        try:
            datetime.strptime(args[4], '%Y-%m-%dT%H:%M:%S').timestamp()
        except Exception as e:
            print('Error parsing date: ' + args[1])
            print_usage()
            return
        #Test validity of end date string
        try:
            datetime.strptime(args[5], '%Y-%m-%dT%H:%M:%S').timestamp()
        except Exception as e:
            print('Error parsing date: ' + args[2])
            PrintUsage()
            return
        start = args[4]
        end = args[5]

    print("api_client_id: {}".format(api_client_id))
    print("api_client_secret: {}".format(api_client_secret))
    print("start: {}".format(start))
    print("start: {}".format(start))
    print("end: {}".format(end))
    print("pred_source: {}".format(pred_source))
    print("market: {}".format(market))
    print("currency_pair: {}".format(currency_pair))
    print("period: {}".format(period))
    print("\n")

    #Everything looks okay; proceed with program
    #Grab the data frame
    D = fetch_data_from_api(api_client_id, api_client_secret, market, currency_pair, period)

    # print("D: {}".format(D))

    #The number of previous days of data used
    #when making a prediction
    num_past_days = 16

    plot_data(D)

    #Number of neurons in the input layer
    i = num_past_days * 7 + 1
    #Number of neurons in the output layer
    o = D.shape[1] - 1
    #Number of neurons in the hidden layers
    h = int((i + o) / 2)
    #The list of layer sizes
    layers = [i, h, h, h, h, h, h, o]

    if(pred_source.startswith('K_NEIGHBORS')):
        R = KNeighborsRegressor(n_neighbors = 5)
    elif(predSource.startswith('MLPR')):
        R = MLPR(layers, maxItr = 1000, tol = 0.60, reg = 0.001, verbose = False)
    else:
        print("Source not implemented yet!")
        print_usage()
        return

    sp = Predictor(R, nPastDays = num_past_days)
    #Learn the dataset and then display performance statistics
    sp.Learn(D)
    sp.TestPerformance()
    #Perform prediction for a specified date range

    P = sp.PredictDate(start, end, period)
    print("P.shape[0]: {}".format(P.shape[0]))
    # print("P: {}".format(P))

    #Keep track of number of predicted results for plot
    n = P.shape[0]
    #Append the predicted results to the actual results
    D = P.append(D)

    #Predicted results are the first n rows
    plot_data(D, range(n + 1))

    print("\n");
    print("End time = %s" % datetime.now())
    print("-------------------------------------------------------------------")

    return (P, n)


#Main entry point for the program
if __name__ == "__main__":
    p, n = main(sys.argv[1:])
