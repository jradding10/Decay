'''
DECAY PREDICTION
Takes user input to predict the discharge energy (Wh). 
'''
import sys
import copy
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import joblib
from decay_ml import ml_battery_dt


if __name__ == "__main__":
    #battery_data = sys.argv[1]
    cycle_num = sys.argv[1]
    test_t = sys.argv[2]
    min_v = sys.argv[3]
    model = joblib.load('battery-life.joblib')
    print(model.predict([[float(cycle_num), float(test_t),
                        float(min_v)]]))
