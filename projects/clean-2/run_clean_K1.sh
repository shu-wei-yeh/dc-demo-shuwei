#!/bin/bash

nohup python clean.py \
    --config config.yaml \
    --data.fname /home/shuwei.yeh/deepclean/data/output.hdf5 \
    --clean.logger.save_dir /home/shuwei.yeh/deepclean/results/K1_clean > K1_clean.log 2>&1 &