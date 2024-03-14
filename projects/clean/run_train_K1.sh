#!/bin/bash

nohup poetry run python train \
    --config config_K1.yaml \
    --data.fname /home/shuwei.yeh/deepclean/data/K-K1_lldata-1369291863-12288.hdf5 \
    --trainer.logger.save_dir /home/shuwei.yeh/deepclean/results/K1_train_test_default > K1_train_test_default.log 2>&1 &
