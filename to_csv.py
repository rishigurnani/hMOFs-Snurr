import sys
import shutil as s

def to_csv(args):
	if len(args)==2:
		suffix=args[1]
	else:
		suffix=''
	arg = args[0]
	f = arg.split('.')[0]
	s.copyfile(arg, f+suffix+'.csv')

to_csv(sys.argv[1:])
