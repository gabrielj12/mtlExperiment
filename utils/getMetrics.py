from sklearn import metrics
import numpy as np
import math



def getRegrMetrics(y_real,y_pred):
    scores = {}
    scores["mse"] = metrics.mean_squared_error(y_real,y_pred)
    scores["rmse"] = math.sqrt(scores["mse"])
    scores["mae"] = metrics.mean_absolute_error(y_real,y_pred)
    scores["r2"] = metrics.r2_score(y_real,y_pred)

    avaiable_metrics = scores.keys()

    return (scores)

def  getClassifMetrics(y_real, y_pred):
    scores = {}
    binary_task = True if len(set(y_real)) == 2 else False

    scores["accuracy"] = metrics.accuracy_score(y_real,y_pred)
    try:
        scores["kappa"] = metrics.cohen_kappa_score(y_real,y_pred)
        scores["hamming_loss"] = metrics.hamming_loss(y_real,y_pred)

        if binary_task:
            scores["precision"] = metrics.precision_score(y_real,y_pred,average="binary")
            scores["f1_score"] = metrics.f1_score(y_real,y_pred,average="binary")
            scores["recall"] = metrics.recall_score(y_real,y_pred,average="binary")
        else:
            scores["avg_precision"] = np.mean(metrics.precision_score(y_real,y_pred,average=None))
            scores["avg_f1_score"] = np.mean(metrics.f1_score(y_real,y_pred,average=None))
            scores["avg_recall"] = np.mean(metrics.recall_score(y_real,y_pred,average=None))

            scores["weighted_precision"] = metrics.precision_score(y_real,y_pred,average="weighted")
            scores["weighted_f1_score"] = metrics.f1_score(y_real,y_pred,average="weighted")
            scores["weighted_recall"] = metrics.recall_score(y_real,y_pred,average="weighted")

            scores["macro_precision"] = metrics.precision_score(y_real,y_pred,average="macro")
            scores["macro_f1_score"] = metrics.f1_score(y_real,y_pred,average="macro")
            scores["macro_recall"] = metrics.recall_score(y_real,y_pred,average="macro")

            scores["micro_precision"] = metrics.precision_score(y_real,y_pred,average="micro")
            scores["micro_f1_score"] = metrics.f1_score(y_real,y_pred,average="micro")
            scores["micro_recall"] = metrics.recall_score(y_real,y_pred,average="micro")
    except:
        pass


    return (scores)