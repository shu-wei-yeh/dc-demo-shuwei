#!/bin/bash

for ifo in H1 L1 K1;
do
    pkill -f "bash replay_${ifo}.sh"
    pkill -f "python replay.py *"
done