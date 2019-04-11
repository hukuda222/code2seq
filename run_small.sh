#!/bin/bash

python main.py --trainpath /home/lr/fujisyo/java-small/output.h5 -g 0 --validpath /home/lr/fujisyo/java-small/output_valid.h5  --datapath /home/lr/fujisyo/java-small/java-small.dict.c2s -e 20 -b 420 --savename small --trainnum 691974 --validnum 23844
