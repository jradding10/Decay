# Decay
Decay is a decision tree regression machine learning algorithm that can be used to find the degrading discharge energy (Wh) of a battery cell.  

Use https://www.batteryarchive.org/list.html?t=0001 and find a battery cell CSV file you would like to analyze. Choose the cycle data download after choosing a cell. Decay works with any battery data chosen from BatteryArchive.org. 

# Set up
Follow the steps in the following video: 

<iframe width="560" height="315" src="https://www.youtube.com/embed/VJwiJiQmS4U" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

1. Download cycle data from https://www.batteryarchive.org/list.html?t=0001 
2. Use decay_ml.py to find the RMSE and get a graphic representation of the decision tree in dot format. 
Use preferred VIM: `py decay_ml.py battery-cell-file-name.csv`
3. Use decay_ml_trainer.py to store the machine learning model in a joblib file. 
Use preferred VIM: `py decay_ml_trainer.py battery-cell-file-name.csv`
4.  Use decay_ml_prediction.py to enter in cycle index, test time, and minimum voltage to find the predicted discharge energy. 
Use preferred VIM: `py decay_ml_prediction.py cycle-index test-time min-voltage`

# Requirements
Libraries needed: scikit-learn, joblib, and pandas 

scikit-learn: https://scikit-learn.org/stable/install.html

joblib: https://pypi.org/project/joblib/

pandas: https://pypi.org/project/joblib/
