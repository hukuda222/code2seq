#!/bin/bash

python valid.py --trainpath /home/lr/fujisyo/java-small/output.h5 -g 0 --validpath /home/lr/fujisyo/java-small/output.h5  --datapath /home/lr/fujisyo/java-small/java-small.dict.c2s -e 1 -b 30 --resume small2.model --trainnum 691974 --validnum 100
#-b 450 --resume small5.model --trainnum 691974 --validnum 23844
