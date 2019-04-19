#!/bin/bash

python main.py --trainpath train.h5 -g 0 --validpath valid.h5 --datapath java-large.dict.c2s -e 40 -b 256 --savename output_seq --trainnum 15344512 --validnum 320866
