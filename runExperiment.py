import configparser
from sklearn.model_selection import cross_val_predict
from utils.getLearner import getClassifier, getRegressor
from utils.readData import getData
from utils.getMetrics import getClassifMetrics
from utils.resampleClassif import resampleClassif
from utils.resampleRegr import resampleRegr

classification = True

configParser = configparser.RawConfigParser()
configParser.read("expConfig.config")

name = configParser.get('EXPERIMENT',"name")
algorithm = configParser.get('EXPERIMENT','algorithm')
data_file = configParser.get('EXPERIMENT','data')
resample_tech = configParser.get('EXPERIMENT','resample_tech')

data_file = "./data/"+data_file+".csv"

X,y = getData(data_file)


if(len(y.columns)>1):
    print ("Multitarget not supported yet!")
    exit()


if(algorithm.startswith("regr.")): classification = False
model = getClassifier(algorithm) if classification else getRegressor(algorithm)

task = name+"."+algorithm+"."+resample_tech

validationTask = resampleClassif(task,model,resample_tech,X,y,save_models=True) if classification \
    else resampleRegr(task,model,resample_tech,X,y,save_models=True)

validationTask.evaluate()
