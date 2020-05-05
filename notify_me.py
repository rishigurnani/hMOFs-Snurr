import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--cmd", type=str,
                    help="command to execute")
parser.add_argument("--subject", type=str, help='Subject line')
parser.add_argument("--cuda", type=str, default='t', help='Command should look for specific CUDA device')
parser.add_argument("--devices", type=str, default='1', help='Specific CUDA device to look for')
parser.add_argument("--address", type=str, help='Email address to notify', default='rgurnani96@gatech.edu')

args = parser.parse_args()

cuda = args.cuda.lower()

if cuda=='t':
    cuda_str = 'CUDA_VISIBLE_DEVICES=%s ' % args.devices
elif cuda=='f':
    cuda_str = '' 
else:
    raise ValueError("invalid cuda flag...choose 't' or 'f'.")

os.system('%s%s && echo "Subject:%s" | sendmail %s' %(cuda_str, args.cmd, args.subject, args.address) )
