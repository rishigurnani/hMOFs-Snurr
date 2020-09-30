import pandas as pd
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--compress", type=str,
                    help="list of files to compress", nargs='+')
#args = sys.argv
args = parser.parse_args()
#path = args[1]
for path in args.compress:
    print('Compressing %s' %path)
    sys.stdout.flush()
    df = pd.read_csv(path)

    df.to_csv(path, compression='gzip')