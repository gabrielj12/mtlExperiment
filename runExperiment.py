import configparser
from utils.getLearner import getClassifier, getRegressor

classification = True

configParser = configparser.RawConfigParser()
configParser.read("expConfig.config")

algorithm = configParser.get('EXPERIMENT','algorithm')
if(algorithm.startswith("regr.")): classification = False

model = getClassifier() if classification else getRegressor()

