#!/bin/bash

python main.py --trainpath /raid/fujisyo/java-large/java-large-train-split/split -g 0 --validpath /raid/fujisyo/java-large/java-large-val-split/split   --datapath /raid/fujisyo/java-large/java-large.dict.c2s -e 20 -b 450 --savename output_seq --trainnum 15344512 --validnum 320866,
