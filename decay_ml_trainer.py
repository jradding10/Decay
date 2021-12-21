'''
DECAY MODEL TRAINING
Saves regression data tree model 
'''
import sys
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import joblib
from decay_ml import ml_battery_dt


def job_maker(dt):
    ''' stores trained model
    Arguments:
    dt: the regression data tree
    returns:
    'battery-life.joblib': a trained regression dt model
    
    '''
    joblib.dump(dt, 'battery-life.joblib')


if __name__ == "__main__":
    battery_data = sys.argv[1]
    dt = ml_battery_dt(battery_data)
    print(type(dt))
    job_maker(dt)
        
