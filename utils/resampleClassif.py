from sklearn.model_selection import KFold
from utils.getMetrics import getClassifMetrics
from utils.fileWriter import resultsToFile
import pandas as pd
import os
import pickle


class resampleClassif:
    __final_metrics = {}
    __predictions = []
    __models = []

    def __init__(self, task_id, clf, r_method, data_x, data_y, save_models=False, output_dir="output/",log_file="log.txt"):
        self.__clf = clf
        self.__task_id = task_id
        self.__rmethod = r_method
        self.__data_x = data_x
        self.__data_y = data_y
        self.__save_models = save_models
        self.__output_dir = output_dir
        self.__log = log_file
        os.makedirs(os.path.join(self.__output_dir,self.__task_id), exist_ok=True)
        with open(os.path.join(self.__output_dir,self.__task_id,self.__log),"w") as f:
                f.write("Starting task {}...\n".format(self.__task_id))


    def __kfold_cross_validation (self, k = 5):
        kf = KFold(n_splits=k)
        id_fold = 1
        predicted_values = pd.DataFrame()
        if self.__save_models: os.makedirs(os.path.join(self.__output_dir,self.__task_id,"models"), exist_ok=True)
        by_fold_test = []
        by_fold_train = []
        for train, test in kf.split(self.__data_x,y=self.__data_y):
            X_train, X_test, Y_train, Y_test = self.__data_x.iloc[train,:], self.__data_x.iloc[test,:], self.__data_y.iloc[train,:], self.__data_y.iloc[test,:]
            print ("Training fold {}...".format(id_fold))
            self.__clf.fit(X_train.drop(["id"],axis=1),Y_train.values.ravel())
            predict_train = self.__clf.predict(X_train.drop(["id"],axis=1))
            metrics_train = getClassifMetrics(Y_train,predict_train)
            print ("Testing fold {}...".format(id_fold))
            predict_test = self.__clf.predict(X_test.drop(["id"],axis=1))
            metrics_test = getClassifMetrics(Y_test, predict_test)
            predicted_values = pd.concat([predicted_values, pd.concat([X_test["id"].reset_index(drop=True), pd.DataFrame(predict_test) ],axis=1)], axis = 0)
            by_fold_test.append([id_fold,metrics_test])
            by_fold_train.append([id_fold,metrics_train])
            self.__models.append(self.__clf)
            if self.__save_models:
                with open(os.path.join(self.__output_dir,self.__task_id)+"/models/model_fold_{}.pickle".format(id_fold), 'wb') as handle:
                    pickle.dump(self.__clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
            id_fold = id_fold + 1
        predicted_values.columns = ['id','predicted']
        self.__predictions = predicted_values
        self.__final_metrics = getClassifMetrics(y_real=self.__data_y, y_pred=self.__predictions["predicted"])
        with open(os.path.join(self.__output_dir,self.__task_id,self.__log),"a+") as f:
            f.write("\n Train Statistics \n")
        resultsToFile(os.path.join(self.__output_dir,self.__task_id,self.__log), by_fold_train)
        with open(os.path.join(self.__output_dir,self.__task_id,self.__log),"a+") as f:
            f.write("\n Test Statistics \n")
        resultsToFile(os.path.join(self.__output_dir,self.__task_id,self.__log), by_fold_test)
        with open(os.path.join(self.__output_dir,self.__task_id,self.__log),"a+") as f:
            f.write("\n Final Statistics \n")
        resultsToFile(os.path.join(self.__output_dir,self.__task_id,self.__log), self.__final_metrics, last_metric=True)

        #with open(,"a+") as f:
        #        f.write("Final Metrics {}\n".format(self.__final_metrics))


    def __holdout():
        return 1

    def __loo(self):
        kf = KFold(n_splits=len(self.__data_x.index))
        id_fold = 1
        predicted_values = pd.DataFrame()
        if self.__save_models: os.makedirs(os.path.join(self.__output_dir,self.__task_id,"models"), exist_ok=True)
        by_fold_test = []
        by_fold_train = []
        for train, test in kf.split(self.__data_x,y=self.__data_y):
            X_train, X_test, Y_train, Y_test = self.__data_x.iloc[train,:], self.__data_x.iloc[test,:], self.__data_y.iloc[train,:], self.__data_y.iloc[test,:]
            print ("Training fold {}...".format(id_fold))
            self.__clf.fit(X_train.drop(["id"],axis=1),Y_train.values.ravel())
            predict_train = self.__clf.predict(X_train.drop(["id"],axis=1))
            metrics_train = getClassifMetrics(Y_train,predict_train)
            print ("Testing fold {}...".format(id_fold))
            predict_test = self.__clf.predict(X_test.drop(["id"],axis=1))
            metrics_test = getClassifMetrics(Y_test,predict_test)
            predicted_values = pd.concat([predicted_values, pd.concat([X_test["id"].reset_index(drop=True), pd.DataFrame(predict_test) ],axis=1)], axis = 0)
            by_fold_test.append([id_fold,metrics_test])
            by_fold_train.append([id_fold,metrics_train])
            if self.__save_models:
                with open(os.path.join(self.__output_dir,self.__task_id,"models")+"/model_fold_{}.pickle".format(id_fold), 'wb') as handle:
                    pickle.dump(self.__clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
            id_fold = id_fold + 1
        predicted_values.columns = ['id','predicted']
        self.__predictions = predicted_values
        self.__final_metrics = getClassifMetrics(y_real=self.__data_y,y_pred= self.__predictions["predicted"])
        with open(os.path.join(self.__output_dir,self.__task_id,self.__log),"a+") as f:
            f.write("\n Train Statistics \n")
        resultsToFile(os.path.join(self.__output_dir,self.__task_id,self.__log), by_fold_train)
        with open(os.path.join(self.__output_dir,self.__task_id,self.__log),"a+") as f:
            f.write("\n Test Statistics \n")
        resultsToFile(os.path.join(self.__output_dir,self.__task_id,self.__log), by_fold_test)
        with open(os.path.join(self.__output_dir,self.__task_id,self.__log),"a+") as f:
            f.write("\n Final Statistics \n")
        resultsToFile(os.path.join(self.__output_dir,self.__task_id,self.__log), self.__final_metrics, last_metric=True)

    def evaluate(self):
        if self.__rmethod == "LOO":
            print ("Evaluating through LOO")
            self.__loo()
        else:
            print ("Evaluating through k-Fold CV")
            folds = int(self.__rmethod[:-3])
            self.__kfold_cross_validation(k=folds)
    def getMetrics(self):
        return (self.__final_metrics)
    def getModels(self):
        return (self.__models)