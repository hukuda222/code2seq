import h5py
from numpy import random as rnd
import pickle
import tqdm

index = 0
# /raid/fujisyo/java-large/java-large.train.c2s"
with h5py.File("/raid/fujisyo/poyo.h5", "w") as h5f:
    with open("/raid/fujisyo/java-large/java-large.val.c2s", "r") as cf:
        for line in tqdm.tqdm(cf):
            h5f.create_group(str(index))
            h5f[str(index)].create_dataset('line', data=line)
            index += 1
