#!/bin/bash

nohup poetry run python -m train \
    --config config_H1_test.yaml \
    --data.fname /home/shuwei.yeh/deepclean/data/H-H1_lldata-1369341015-12288.hdf5 \
    --trainer.logger.save_dir /home/shuwei.yeh/deepclean/results/H1_train_test > H1_train_test.log 2>&1 &