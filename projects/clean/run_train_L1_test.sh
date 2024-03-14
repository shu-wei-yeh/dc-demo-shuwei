#!/bin/bash

nohup poetry run python -m train \
    --config config_L1_test.yaml \
    --data.fname /home/shuwei.yeh/deepclean/data/L-L1_lldata-1369341015-12288.hdf5 \
    --trainer.logger.save_dir /home/shuwei.yeh/deepclean/results/L1_train_test > L1_train_test.log 2>&1 &