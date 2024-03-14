#!/bin/bash
#set -x

HOFT_SOURCE=/home/chiajui.chou/ll_data/llhoft_buffer
HOFT_DESTINATION=/home/shuwei.yeh/Data_Gringotts_Bank/online_dc_data/ll_replay/data/kafka
WITNESS_SOURCE=/home/chiajui.chou/ll_data/lldetchar_buffer
WITNESS_DESTINATION=/home/shuwei.yeh/Data_Gringotts_Bank/online_dc_data/ll_replay/data/lldetchar
START=1369291863
DURATION=16384
KEEP=300

function start_replay {
    ifo=$1
    hoft_source=${HOFT_SOURCE}/${ifo}
    hoft_destination=${HOFT_DESTINATION}/${ifo}
    witness_source=${WITNESS_SOURCE}/${ifo}
    witness_destination=${WITNESS_DESTINATION}/${ifo}
    python_params=(
        replay.py
        --ifo ${ifo}
        --hoft_source ${hoft_source}
        --witness_source ${witness_source}
        --hoft_destination ${hoft_destination}
        --witness_destination ${witness_destination}
        --start ${START}
        --duration ${DURATION}
        --keep ${KEEP}
    )
    rm ${hoft_destination}/*
    rm ${witness_destination}/*
    python "${python_params[@]}"
}

# start_replay H1 &> replay_H1.out &
# start_replay L1 &> replay_L1.out &
start_replay K1 &> replay_K1.out &
