#!/usr/bin/bash
rm -rf /home/nfs/zsc/yuan/coldata_dbg/dataset/012
rm /home/nfs/zsc/yuan/coldata_dbg/dataset/data_list.pt
ps -aux | grep colossalai | awk '{print $2}' | xargs kill