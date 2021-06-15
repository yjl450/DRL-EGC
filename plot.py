from re import L
from parse import *
from parse import compile
import argparse
import seaborn as sns
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

sns.set()

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str)
parser.add_argument("--algo", type=int)
args = parser.parse_args()
log_dir = args.logdir

# peval = compile("INFO:__main__:evaluation episode {} length:{} R:{:g} AWT: {}")

train_return=[]

if args.algo == 1:
    ptrain = compile("INFO:__main__:outdir:{} step:{} episode:{} R:{}")
else:
    ptrain = compile("INFO:__main__:outdir:{} global_step:{} local_step:{} R:{}")
    # ptrain = compile("INFO:__main__:outdir:{} step:{} episode:{} R:{}")
with tqdm(total=os.path.getsize(log_dir)) as pbar:
    with open(log_dir,'r') as file_:
        for line in tqdm(file_):
            pbar.update(len(line))
            p1 = ptrain.parse(line)
            # p2 = peval.parse(line)
            if p1:
                train_return.append(float(p1[3]))
                index = int(p1[1])
                if int(p1[1]) == 9078711:
                    break




print(len(train_return), index)

# sns.lineplot(x="timepoint", y="signal",hue="region", style="event", data=np.array(train_return))
# plt.plot(data=np.array(train_return))
# print(len(train_return), len(list(range(index//1000))))
sns.lineplot(range(int(len(train_return))),train_return)
plt.xlabel("Time Step (Thousand Steps)")
plt.ylabel("Return")
plt.show()