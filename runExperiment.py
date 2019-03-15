import configparser
from sklearn.model_selection import cross_val_predict
from utils.getLearner import getClassifier, getRegressor
from utils.readData import getData
from utils.getMetrics import getClassifMetrics

classification = True

configParser = configparser.RawConfigParser()
configParser.read("expConfig.config")

algorithm = configParser.get('EXPERIMENT','algorithm')
data_file = configParser.get('EXPERIMENT','data')
resample_techs = configParser.get('EXPERIMENT','resample_tech')



data_file = "./data/"+data_file+".csv"

X,y = getData(data_file)

if resample_techs == "LOO":
    folds = X.shape[0]
else:
    folds = int(resample_techs[:-3])


if(len(y.columns)>1):
    print ("Multitarget not supported yet!")
    exit()


if(algorithm.startswith("regr.")): classification = False
model = getClassifier(algorithm) if classification else getRegressor(algorithm)

#print(y.shape)

y_pred = cross_val_predict(model, X.drop(['id'], axis=1), y.values.ravel(), cv=folds)

print (getClassifMetrics(y,y_pred,metrics_list=["accuracy","cohen"]))
