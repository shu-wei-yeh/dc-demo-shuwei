#!/bin/bash

nohup poetry run python train \
    --config config_L1.yaml \
    --data.fname /home/shuwei.yeh/deepclean/data/L-L1_lldata-1369291863-12288.hdf5 \
    --trainer.logger.save_dir /home/shuwei.yeh/deepclean/results/L1_train_test_dc-prod > L1_train_test_dc-prod.log 2>&1 &