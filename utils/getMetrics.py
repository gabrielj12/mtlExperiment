from sklearn import metrics


def  getMetrics(y_real, y_pred, metrics_list = ['accuracy']):
    scores = {}
    scores["accuracy"] = metrics.accuracy_score(y_real,y_pred)
    scores["precision"] = metrics.precision_score(y_real,y_pred,average='weighted')
    return (scores)