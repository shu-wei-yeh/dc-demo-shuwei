#!/bin/bash
IFO=K1
HOFT_SOURCE=/home/chiajui.chou/ll_data/llhoft_buffer/${IFO}
HOFT_DESTINATION=/home/chiajui.chou/ll_data/data/kafka/${IFO}
WITNESS_SOURCE=/home/chiajui.chou/ll_data/lldetchar_buffer/${IFO}
WITNESS_DESTINATION=/home/chiajui.chou/ll_data/data/lldetchar/${IFO}
START=1369291863
DURATION=16384
KEEP=300

rm ${HOFT_DESTINATION}/*
rm ${WITNESS_DESTINATION}/*

python replay.py\
	--ifo ${IFO}\
	--hoft_source ${HOFT_SOURCE}\
	--witness_source ${WITNESS_SOURCE}\
	--hoft_destination ${HOFT_DESTINATION}\
	--witness_destination ${WITNESS_DESTINATION}\
	--start ${START}\
	--duration ${DURATION}\
	--keep ${KEEP}\
