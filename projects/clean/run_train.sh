#!/bin/bash

nohup poetry run python train \
    --config config.yaml \
    --data.fname /home/shuwei.yeh/deepclean/data/K-K1_lldata-1369291863-16384.hdf5 \
    --trainer.logger.save_dir /home/shuwei.yeh/deepclean/results/K1_training_test_3 > training_output_3.log 2>&1 &
