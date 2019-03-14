from sklearn import metrics


def  getMetrics(y_real, y_pred, metrics_list = ['accuracy']):
    scores = {}
    scores["accuracy"] = metrics.accuracy_score(y_real,y_pred)

    #scores["precision"] = metrics.precision_score(y_real,y_pred,average="binary")
    #scores["f1_score"] = metrics.f1_score(y_real,y_pred,average="binary")
    #scores["recall"] = metrics.recall_score(y_real,y_pred,average="binary")

    scores["avg_precision"] = metrics.precision_score(y_real,y_pred,average=None)
    scores["avg_f1_score"] = metrics.f1_score(y_real,y_pred,average=None)
    scores["avg_recall"] = metrics.recall_score(y_real,y_pred,average=None)

    scores["weighted_precision"] = metrics.precision_score(y_real,y_pred,average="weighted")
    scores["weighted_f1_score"] = metrics.f1_score(y_real,y_pred,average="weighted")
    scores["weighted_recall"] = metrics.recall_score(y_real,y_pred,average="weighted")

    scores["macro_precision"] = metrics.precision_score(y_real,y_pred,average="macro")
    scores["macro_f1_score"] = metrics.f1_score(y_real,y_pred,average="macro")
    scores["macro_recall"] = metrics.recall_score(y_real,y_pred,average="macro")

    scores["micro_precision"] = metrics.precision_score(y_real,y_pred,average="micro")
    scores["micro_f1_score"] = metrics.f1_score(y_real,y_pred,average="micro")
    scores["micro_recall"] = metrics.recall_score(y_real,y_pred,average="micro")

    scores["kappa"] = metrics.cohen_kappa_score(y_real,y_pred)
    scores["hamming_loss"] = metrics.hamming_loss(y_real,y_pred)
    return (scores)