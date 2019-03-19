from sklearn.model_selection import KFold
from utils.getMetrics import getRegrMetrics
import pandas as pd
import os
import pickle


class resampleRegr:
    __final_metrics = {}
    __predictions = []
    __models = []

    def __init__(self, task_id, clf, r_method, data_x, data_y, metric_list=["mse"], save_models=False, output_dir="output/models/"):
        self.__clf = clf
        self.__task_id = task_id
        self.__rmethod = r_method
        self.__metric_list = metric_list
        self.__data_x = data_x
        self.__data_y = data_y
        self.__save_models = save_models
        self.__output_dir = output_dir

    def __kfold_cross_validation (self, k = 5):
        kf = KFold(n_splits=k)
        id_fold = 1
        predicted_values = pd.DataFrame()
        if self.__save_models: os.makedirs(os.path.join(self.__output_dir,self.__task_id), exist_ok=True)
        for train, test in kf.split(self.__data_x,y=self.__data_y):
            X_train, X_test, Y_train, Y_test = self.__data_x.iloc[train,:], self.__data_x.iloc[test,:], self.__data_y.iloc[train,:], self.__data_y.iloc[test,:]
            self.__clf.fit(X_train.drop(["id"],axis=1),Y_train.values.ravel())
            predict_train = self.__clf.predict(X_train.drop(["id"],axis=1))
            mse_train = getRegrMetrics(Y_train,predict_train,metrics_list=["mse"])
            predict_test = self.__clf.predict(X_test.drop(["id"],axis=1))
            mse_test = getRegrMetrics(Y_test, predict_test,metrics_list=["mse"])
            predicted_values = pd.concat([predicted_values, pd.concat([X_test["id"].reset_index(drop=True), pd.DataFrame(predict_test) ],axis=1)], axis = 0)
            print ("Fold {} MSE Train.{} MSE Test.{}".format(id_fold,mse_train,mse_test))
            self.__models.append(self.__clf)
            if self.__save_models:
                with open(os.path.join(self.__output_dir,self.__task_id)+"/model_fold_{}.pickle".format(id_fold), 'wb') as handle:
                    pickle.dump(self.__clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
            id_fold = id_fold + 1
        predicted_values.columns = ['id','predicted']
        self.__predictions = predicted_values
        self.__final_metrics = getRegrMetrics(y_real=self.__data_y, y_pred=self.__predictions["predicted"])

    def __holdout():
        return 1

    def __loo(self):
        kf = KFold(n_splits=len(self.__data_x.index))
        id_fold = 1
        predicted_values = pd.DataFrame()
        if self.__save_models: os.makedirs(os.path.join(self.__output_dir,self.__task_id), exist_ok=True)
        for train, test in kf.split(self.__data_x,y=self.__data_y):
            X_train, X_test, Y_train, Y_test = self.__data_x.iloc[train,:], self.__data_x.iloc[test,:], self.__data_y.iloc[train,:], self.__data_y.iloc[test,:]
            self.__clf.fit(X_train.drop(["id"],axis=1),Y_train.values.ravel())
            predict_train = self.__clf.predict(X_train.drop(["id"],axis=1))
            mse_train = getRegrMetrics(Y_train,predict_train,metrics_list=["mse"])
            predict_test = self.__clf.predict(X_test.drop(["id"],axis=1))
            mse_test = getRegrMetrics(Y_test, predict_test,metrics_list=["mse"])
            predicted_values = pd.concat([predicted_values, pd.concat([X_test["id"].reset_index(drop=True), pd.DataFrame(predict_test) ],axis=1)], axis = 0)
            print ("Fold {} MSE Train.{} MSE Test.{}".format(id_fold,mse_train,mse_test))
            if self.__save_models:
                with open(os.path.join(self.__output_dir,self.__task_id)+"/model_fold_{}.pickle".format(id_fold), 'wb') as handle:
                    pickle.dump(self.__clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
            id_fold = id_fold + 1
        self.__predictions = predicted_values
        self.__final_metrics = getRegrMetrics(self.__data_y, self.__predictions)

    def evaluate(self):
        if self.__rmethod == "LOO":
            self.__loo()
        else:
            folds = int(self.__rmethod[:-3])
            self.__kfold_cross_validation(k=folds)
    def getMetrics(self):
        return (self.__final_metrics)
    def getModels(self):
        return (self.__models)