import h5py
from numpy import random as rnd
import pickle
import tqdm

index = 0
with h5py.File("train.h5", "w") as h5f:
    with open("java-small.train.c2s", "r") as cf:
        for line in tqdm.tqdm(cf):
            h5f.create_group(str(index))
            h5f[str(index)].create_dataset('line', data=line)
            index += 1
