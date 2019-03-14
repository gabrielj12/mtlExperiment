import pandas as pd
import numpy as np


def getData(data_file):
    dataset = pd.read_csv(data_file)
    class_columns = [col for col in dataset.columns if col.startswith("class")]
    print (class_columns)
    return ([1,2],[2])