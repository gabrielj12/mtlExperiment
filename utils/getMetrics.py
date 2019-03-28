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

    scores["acc"] = metrics.accuracy_score(y_real,y_pred)
    try:
        scores["kappa"] = metrics.cohen_kappa_score(y_real,y_pred)
        scores["hm_loss"] = metrics.hamming_loss(y_real,y_pred)

        if binary_task:
            scores["prec"] = metrics.precision_score(y_real,y_pred,average="binary")
            scores["f1"] = metrics.f1_score(y_real,y_pred,average="binary")
            scores["rcll"] = metrics.recall_score(y_real,y_pred,average="binary")
        else:
            scores["avg_prec"] = np.mean(metrics.precision_score(y_real,y_pred,average=None))
            scores["avg_f1"] = np.mean(metrics.f1_score(y_real,y_pred,average=None))
            scores["avg_rcll"] = np.mean(metrics.recall_score(y_real,y_pred,average=None))

            scores["wgh_prec"] = metrics.precision_score(y_real,y_pred,average="weighted")
            scores["wgh_f1"] = metrics.f1_score(y_real,y_pred,average="weighted")
            scores["wgh_rcll"] = metrics.recall_score(y_real,y_pred,average="weighted")

            scores["mac_prec"] = metrics.precision_score(y_real,y_pred,average="macro")
            scores["mac_f1"] = metrics.f1_score(y_real,y_pred,average="macro")
            scores["mac_rcll"] = metrics.recall_score(y_real,y_pred,average="macro")

            scores["mic_prec"] = metrics.precision_score(y_real,y_pred,average="micro")
            scores["mic_f1"] = metrics.f1_score(y_real,y_pred,average="micro")
            scores["mic_rcll"] = metrics.recall_score(y_real,y_pred,average="micro")
    except:
        pass


    return (scores)