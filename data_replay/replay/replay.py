#!/bin/python

import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ifo", default=None, help="The Name of the detector.")
parser.add_argument("-hs", "--hoft_source", default=None, help="The directory of the gwf files of the hoft source.")
parser.add_argument("-ws", "--witness_source", default=None, help="The directory of the gwf files of the witness source.")
parser.add_argument("-hd", "--hoft_destination", default=None, help="The directory of the destination of the hoft gwf files.")
parser.add_argument("-wd", "--witness_destination", default=None, help="The directory of the destination of the witness gwf files.")
parser.add_argument("-s", "--start", default=None, type=int, help="The GPS time of the start of the replay.")
parser.add_argument("-d", "--duration", default=None, type=int, help="The duration of the replay.")
parser.add_argument("-k", "--keep", default=None, type=int, help="The number of the gwf files to keep in the destination directories.")
args = parser.parse_args()

replay_start = 1369291863
duration = 16384
keep = 300

ifo = "K1"
hoft_src = f"/data/ll_data/llhoft_buffer/{ifo}"
woft_src = f"/data/ll_data/lldetchar_buffer/{ifo}"
hoft_dest = f"/data/ll_data/kafka/{ifo}"
woft_dest = f"/data/ll_data/lldetchar/{ifo}"

def main():
    ifo = args.ifo
    hoft_src = args.hoft_source
    woft_src = args.witness_source
    hoft_dest = args.hoft_destination
    woft_dest = args.witness_destination
    replay_start = args.start
    duration = args.duration
    keep = args.keep

    # increments by one after writing every frame-set (L1 H1 both hoft and woft)
    total_streaming_time_should_now_be = 0

    print (f"{ifo} Replay starts: {replay_start} s.")
    begin = time.time()
    for start in range(replay_start, replay_start+duration):
        try:
            # strain channel frames
            hoft_from = f"{hoft_src}/{ifo[0]}-{ifo}_llhoft-{start}-1.gwf"
            hoft_to = f"{hoft_dest}/{ifo[0]}-{ifo}_llhoft-{start}-1.gwf"

            # witness channel frames
            woft_from = f"{woft_src}/{ifo[0]}-{ifo}_lldetchar-{start}-1.gwf"
            woft_to = f"{woft_dest}/{ifo[0]}-{ifo}_lldetchar-{start}-1.gwf"

            # copy the frames to destination
            os.system(f"cp {hoft_from} {hoft_to}")
            os.system(f"cp {woft_from} {woft_to}")

            # Delete previous files if the number of the files in the destination is larger than the keep nunber.
            count = start - replay_start
            if count >= keep:
                os.system(f"rm {hoft_dest}/{ifo[0]}-{ifo}_llhoft-{start-keep}-1.gwf")
                os.system(f"rm {woft_dest}/{ifo[0]}-{ifo}_lldetchar-{start-keep}-1.gwf")

            if count == (duration-1):
                print(f"{ifo} Replay ends: {replay_start+duration} s.")

        except KeyboardInterrupt:
            print ("KeyboardInterrupt")

        #except:
            #pass

        total_streaming_time_should_now_be += 1
        total_time_elapsed_so_far = time.time() - begin
        wait  = total_streaming_time_should_now_be - total_time_elapsed_so_far
        time.sleep(wait)

if __name__ == "__main__":
    main()
