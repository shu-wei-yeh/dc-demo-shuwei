# Data Replay
We choose the start GPS time of the replay to be 1369291863 s and the end GPS time is 1369369247 s. The duration is 77384 seconds.
This segment is the longest segment with both H1, L1, K1 are in the science mode in O4a.

The channels of each detector can be found in [chanslist.txt](./get_data/chanslist.txt). The first channel is the low-latency strain, the second channel is the state vector and the rest are the witness channels we are going to use for the noise regression by DeepClean.

## Get low-latency strain channel and witness channels
In [get_data](./get_data), the script [make_subs.sh](./get_data/make_subs.sh) is used to generate the working directories and the sub files to collect the data needed using HTCondor in LIGO-Caltech Computing Cluster. The K1 data is collected in KAGRA Main Data Server.

## Make 1-sec gwf files of low-latency data
In [make_lldata](./make_lldata/), the script [make_lldata_subs.sh](./make_lldata/make_lldata_subs.sh) is used to generate the sorking directories and the sub files to make the 1-sec gwf files of the low-latency strain (labeld as ```llhoft```) and the witness channels (labeled as ```lldetchar```) using HTCondor in LIGO-Caltech Computing Cluster.

## Use the data replay
The 1-sec gwf files are stored in the buffer folders: ```/home/chiajui.chou/ll_data/llhoft_buffer/``` for the strain and ```/home/chiajui.chou/ll_data/lldetchar_buffer/``` for the witness channels.

To start the replay, specify the replay configuration in the [start_replay.sh](./replay/start_replay.sh) and execute this script.
The strain data in ```HOFT_SOURCE``` will be copied into ```HOFT_DESTINATION``` sequentially and the data of witness channels in ```WITNESS_SOURCE``` will be copied into the ```WITNESS_DESTINATION```.
The start time and the duration of the replay are defined by ```START``` and ```DURATION```. The maximum number of the gwf files to keep in the destination folder is specified by ```KEEP```.
The data of ```H1```, ```L1``` and ```K1``` will be replayed together. Comment out the line ```start_replay [IFO] &> replay_[IFO].out &``` if you don't want the data of the detector ```[IFO]``` to be replayed.

To stop the replay, execute [stop_replay.sh](./replay/stop_replay.sh). Comment our the line ```stop_replay [IFO] &``` if you don't want to stop the replay of the detector ```[IFO]```.
