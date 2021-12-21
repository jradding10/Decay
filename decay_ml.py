'''
DECAY DATA ANALYSIS

Created by Jacqueline Radding

Using a regression decision tree, a machine learning
model will be created to predict
the discharge energy (Wh) in a battery cell.
Program returns the RMSE and data tree vistual(.dot). 

Find battery cycle CSV data to use here:
https://www.batteryarchive.org/list.html?time=0001

'''

import sys
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error as MSE


def ml_battery_dt(battery_data):
    '''
    trains the decision tree regression model
    Arguments:
    battery_data(string): user inputed name of csv file
    Returns:
    dt.fit: regression model
    '''
    global X_train, X_test, y_train, y_test
    cell_data = pd.read_csv(battery_data)
    X = cell_data[['Cycle_Index', 'Test_Time (s)', 'Min_Voltage (V)']]
    y = cell_data['Discharge_Energy (Wh)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    dt = DecisionTreeRegressor(max_depth=20, min_samples_leaf=0.1,
                               random_state=3)
    dt.fit(X_train.values, y_train.values) # added values w/o feature
    return dt


def ml_battery_RMS(dt):
    '''s
    Returns:
    rmse_dt(float): root means squared error of data
    source: https://www.youtube.com/watch?v=ksvJDLdc9eA&ab_channel=DataCamp
    '''
    y_pred = dt.predict(X_test.values) # added values without feature
    mse_dt = MSE(y_test, y_pred)
    rmse_dt = mse_dt ** (1/2)
    return rmse_dt


if __name__ == '__main__':
    '''
    Main takes input files from user and outputs RMSE and visual
    '''
    try:
        battery_data = sys.argv[1]  # user selects file
    except IOError:
        print("Unable to open " + battery_data)
        print("Format your entry like this:\npy python-file-name.py csv-file-name.csv")
        exit()

    # train the battery
    dt = ml_battery_dt(battery_data)
    # output visual in dot file
    print("RMSE CALCULATED: " + str(ml_battery_RMS(dt)))
    # output visual in dot file
    out_tree_vis = tree.export_graphviz(dt, out_file='battery_datatree.dot', 
                                feature_names=['Cycle Index', 'Test Time',
                                            'Min Voltage'], filled=True)
    # above source https://mljar.com/blog/visualize-decision-tree/

    

