from sklearn import metrics
import numpy as np





def  getClassifMetrics(y_real, y_pred, metrics_list = ['accuracy']):
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

    avaiable_metrics = scores.keys()
    if not set(metrics_list).issubset(avaiable_metrics):
        print ("Some metrics were not calculed due to the type of the task or are not avaible in this framework\n Avaiable Metrics {}".format(avaiable_metrics))

    for x in list(avaiable_metrics):
        if x not in metrics_list: scores.pop(x, None)

    return (scores)