import pandas as pd
import sys

args = sys.argv

path = args[1]

df = pd.read_csv(path)

df.to_csv(path, compression='gzip')