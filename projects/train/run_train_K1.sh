#!/bin/bash

nohup poetry run python -m train \
    --config config_K1.yaml \
    --data.fname /home/shuwei.yeh/deepclean/data/K-K1_lldata-1370850226-12288.hdf5 \
    --trainer.logger.save_dir /home/shuwei.yeh/deepclean/results/K1_results-new-data > ./log/K1_train-4.log 2>&1 &