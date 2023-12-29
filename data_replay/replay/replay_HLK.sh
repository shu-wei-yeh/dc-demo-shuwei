#!/bin/bash

for ifo in H1 L1 K1
do
    nohup bash replay_${ifo}.sh >> nohup_${ifo}.out &
done
