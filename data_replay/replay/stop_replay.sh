#!/bin/bash

function stop_replay {
    ifo=$1
    pkill -U $USER -f "python replay.py --ifo ${ifo}*"
}

stop_replay H1 &
stop_replay L1 &
stop_replay K1 &