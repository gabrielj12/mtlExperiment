from sklearn.preprocessing import normalize
import pandas as pd

def normalizing(X,y):
    #print (X)
    original_columns = X.columns
    X_noid = pd.DataFrame(normalize(X.drop(["id"],axis=1), axis=0))

    X = pd.concat([X["id"],X_noid.reset_index(drop=True)],axis=1)
    X.columns = original_columns

    y = pd.DataFrame(normalize(y,axis=0))
    y.columns = ["class"]
    return X, y
