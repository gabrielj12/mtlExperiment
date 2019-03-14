import configparser
from utils.getLearner import getClassifier, getRegressor
from utils.readData import getData

classification = True

configParser = configparser.RawConfigParser()
configParser.read("expConfig.config")

algorithm = configParser.get('EXPERIMENT','algorithm')
data_file = configParser.get('EXPERIMENT','data')

data_file = "./data/"+data_file+".csv"

X,y = getData(data_file)


if(algorithm.startswith("regr.")): classification = False
model = getClassifier(algorithm) if classification else getRegressor(algorithm)

