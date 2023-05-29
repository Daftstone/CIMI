import os
from parse import *

args = parse_args(None)

os.system(f"python train_bert.py --device {args.device} --dataset {args.dataset}")
os.system(
    f"python CIMI.py --device {args.device} --dataset {args.dataset} --train_stack --batch_size {args.batch_size}")
os.system(
        f"python eval_main.py --device {args.device} --dataset {args.dataset}")