import pandas as pd
import numpy as np

def init_data(path):
    return pd.read_csv(path)
        
def remove_cols(df, cols):
    df = df.drop(cols, axis=1)
    
    return df
    