#!/bin/bash

nohup poetry run python train \
    --config config_H1.yaml \
    --data.fname /home/shuwei.yeh/deepclean/data/H-H1_lldata-1369291863-12288.hdf5 \
    --trainer.logger.save_dir /home/shuwei.yeh/deepclean/results/H1_train_test_dc-prod > H1_train_test_dc-prod.log 2>&1 &