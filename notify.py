import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--command", type=str,
                    help="command to execute")
parser.add_argument("--subject", type=str, help='Subject line')
parser.add_argument("--address", type=str, help='Email address to notify', default='rgurnani96@gatech.edu')

args = parser.parse_args()

os.system('efrc')
os.system('%s && echo "Subject:%s" | sendmail %s' %(args.command, args.subject, args.address) )