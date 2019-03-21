import configparser
import pandas as pd
from sklearn.model_selection import cross_val_predict

from utils.getLearner import getClassifier, getRegressor
from utils.readData import getData
from utils.getMetrics import getClassifMetrics
from utils.resampleClassif import resampleClassif
from utils.resampleRegr import resampleRegr
from utils.preprocessing import normalizing

classification = True

configParser = configparser.RawConfigParser()
configParser.read("expConfig.config")

name = configParser.get('EXPERIMENT',"name")
algorithm = configParser.get('EXPERIMENT','algorithm')
data_file = configParser.get('EXPERIMENT','data')
resample_tech = configParser.get('EXPERIMENT','resample_tech')
normalize_data = configParser.get('EXPERIMENT','normalize_data')
params = configParser.get('EXPERIMENT', 'hyperparameters')

print (params)

params_model = dict((k.strip(), v.strip()) for k,v in
              (item.split(':') for item in params.split(',')))

for key, value in params_model.items():
    if value.isdigit():
        params_model[key] = int(value)

data_file = "./data/"+data_file+".csv"

X,y = getData(data_file)

if normalize_data == "True": X,y = normalizing(X,y)


if(len(y.columns)>1):
    print ("Multitarget not supported yet!")
    exit()


if(algorithm.startswith("regr.")): classification = False
model = getClassifier(algorithm) if classification else getRegressor(algorithm)


if params != "default": model.set_params(**params_model)


task = name+"."+algorithm+"."+resample_tech

validationTask = resampleClassif(task,model,resample_tech,X,y,save_models=True) if classification \
    else resampleRegr(task,model,resample_tech,X,y,save_models=True)

validationTask.evaluate()
