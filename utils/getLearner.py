from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor

def getClassifier(classif):
    if classif == "classif.randomForest":
        return (RandomForestClassifier(n_estimators=10,n_jobs=-1))
    elif classif == "classif.svm":
        return (SVC(gamma='auto'))
    elif classif == "classif.knn":
        return (KNeighborsClassifier())
    elif classif == "classif.decisionTree":
        return (DecisionTreeClassifier())
    elif classif == "classif.nn":
        return (MLPClassifier())
    elif classif == "classif.xgb":
        return (XGBClassifier())
    else:
        print ("Not expected algorithm.")
        exit()

def getRegressor(regr):
    if regr == "regr.randomForest":
        return (RandomForestRegressor(n_estimators=10,n_jobs=-1))
    elif regr == "regr.svm":
        return (SVR(gamma='auto'))
    elif regr == "regr.knn":
        return (KNeighborsRegressor())
    elif regr == "regr.decisionTree":
        return (DecisionTreeRegressor())
    elif regr == "regr.nn":
        return (MLPClassifier())
    elif regr == "regr.xgb":
        return (XGBRegressor())
    else:
        print ("Not expected algorithm.")
        exit()
