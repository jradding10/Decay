# Decay
Decay is a decision tree regression machine learning algorithm that can be used to find the degrading discharge energy (Wh) of a battery cell.  

Use https://www.batteryarchive.org/list.html?t=0001 and find a battery cell CSV file you would like to analyze. Choose the cycle data download after choosing a cell. 

# Set up
Follow the steps in the following video: https://youtu.be/L0L76S3UVRM
1. Download cycle data from https://www.batteryarchive.org/list.html?t=0001 
2. Use decay_ml.py to find the RMSE and get a graphic representation of the decision tree in dot format. Use preferred VIM: py decay_ml.py battery-cell-file-name.csv
3. Use decay_ml_trainer.py to store the machine learning model in a joblib file. Use preferred VIM: py decay_ml_trainer.py battery-cell-file-name.csv
4.  Use decay_ml_prediction.py to enter in cycle index, test time, and minimum voltage to find the predicted discharge energy. Use preferred VIM: py decay_ml_prediction.py cycle-index test-time min-voltage

# Requirements
Libraries needed: sklearn, joblib, and pandas 

sklearn: https://scikit-learn.org/stable/install.html

joblib: https://pypi.org/project/joblib/

pandas: https://pypi.org/project/joblib/
