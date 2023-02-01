import pandas as pd
import numpy as np

def get_wine():
    return pd.read_csv('wine.csv',index_col=0)