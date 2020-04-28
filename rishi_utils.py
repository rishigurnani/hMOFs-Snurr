import pandas as pd

def pd_load(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.read_csv(path, compression='gzip')