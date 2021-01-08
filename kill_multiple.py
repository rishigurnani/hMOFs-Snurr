import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description='specify test data size or percent')
parser.add_argument("--start", type=int,
                    help="starting pid to kill")
parser.add_argument("--stop", type=int,
                    help="end pid to kill")     

args = parser.parse_args()

for pid in range(args.start, args.stop+1):
    try:
        if subprocess.check_output('ps -o user= -p %s' %pid, shell=True).decode().strip() == 'rgur': #check if pid belongs to me
            os.system('kill %s' %pid)
    except:
        pass