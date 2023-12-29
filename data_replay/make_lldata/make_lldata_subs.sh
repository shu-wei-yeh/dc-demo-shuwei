#!/bin/bash

DATA_DIR=/home/chiajui.chou/ll_data
IFO=L1
#TAG=llhoft
TAG=lldetchar
START=1369291863
END=1369369247
LENGTH=4096
OUTPUT=/home/chiajui.chou/GW-data/ll_replay/make_lldata/${IFO}_${TAG}_condor

python make_lldata_subs.py\
    --ifo ${IFO}\
    --source ${DATA_DIR}/${IFO}_${TAG}\
    --start ${START}\
    --end ${END}\
    --length ${LENGTH}\
    --destination ${DATA_DIR}/${TAG}_buffer/${IFO}\
    --tag ${TAG}\
    --output ${OUTPUT}
