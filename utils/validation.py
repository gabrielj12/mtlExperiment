from sklearn.model_selection import KFold
from utils.getMetrics import getClassifMetrics
import pandas as pd
import os
import pickle


def cross_validation_classification (clf, data_x, data_y, k = 5, task_id="default", save_models = False, output_dir = "output/models/"):
    kf = KFold(n_splits=k)
    id_fold = 1
    predicted_values = pd.DataFrame()
    if save_models: os.makedirs(os.path.join(output_dir,task_id), exist_ok=True)
    for train, test in kf.split(data_x,y=data_y):
        X_train, X_test, Y_train, Y_test = data_x.iloc[train,:], data_x.iloc[test,:], data_y.iloc[train,:], data_y.iloc[test,:]
        clf.fit(X_train.drop(["id"],axis=1),Y_train.values.ravel())
        predict_train = clf.predict(X_train.drop(["id"],axis=1))
        acc_train = getClassifMetrics(Y_train,predict_train,metrics_list=["accuracy"])
        predict_test = clf.predict(X_test.drop(["id"],axis=1))
        acc_test = getClassifMetrics(Y_test, predict_test,metrics_list=["accuracy"])
        predicted_values = pd.concat([predicted_values, pd.concat([X_test["id"].reset_index(drop=True), pd.DataFrame(predict_test) ],axis=1)], axis = 0)
        print ("Fold {} ACC Train.{} ACC Test.{}".format(id_fold,acc_train,acc_test))
        if save_models:
            with open(os.path.join(output_dir,task_id)+"/model_fold_{}.pickle".format(id_fold), 'wb') as handle:
                pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        id_fold = id_fold + 1




def holdout():
    return 1

def loo():
    return 1