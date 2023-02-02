import pandas as pd
import numpy as np

def get_wine():
    df = pd.read_csv('wine.csv',index_col=0)
    df.columns = df.columns.str.replace(' ', '_')
    df.rename(columns={'color':'color_red'}, inplace=True)
    return df