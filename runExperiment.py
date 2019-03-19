import configparser
from sklearn.model_selection import cross_val_predict
from utils.getLearner import getClassifier, getRegressor
from utils.readData import getData
from utils.getMetrics import getClassifMetrics
from utils.validation import cross_validation_classification

classification = True

configParser = configparser.RawConfigParser()
configParser.read("expConfig.config")

name = configParser.get('EXPERIMENT',"name")
algorithm = configParser.get('EXPERIMENT','algorithm')
data_file = configParser.get('EXPERIMENT','data')
resample_tech = configParser.get('EXPERIMENT','resample_tech')



data_file = "./data/"+data_file+".csv"

X,y = getData(data_file)

if resample_tech == "LOO":
    folds = X.shape[0]
else:
    folds = int(resample_tech[:-3])


if(len(y.columns)>1):
    print ("Multitarget not supported yet!")
    exit()


if(algorithm.startswith("regr.")): classification = False
model = getClassifier(algorithm) if classification else getRegressor(algorithm)

task = name+"."+algorithm+"."+resample_tech

print (model)

cross_validation_classification(model,X,y,task_id=task,save_models=True)


#print (model)

#print (model.feature_importances_)

#print(y.shape)

#y_pred = cross_val_predict(model, X.drop(['id'], axis=1), y.values.ravel(), cv=folds)

#print (getClassifMetrics(y,y_pred,metrics_list=["accuracy","cohen"]))
