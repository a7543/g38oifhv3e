#!/usr/bin/bash
export DATA=coldata_dbg/dataset
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

python go.py --config=config.py
#colossalai run --nproc_per_node=1 /home/nfs/zsc/yuan/ColossalAI-Examples/language/gpt/train_gpt.py --config=config.py --from_torch 
echo --hostfile ./hostfile
#python /home/nfs/zsc/yuan/ColossalAI-Examples/language/gpt/train_gpt.py --rank=0 --world_size=1 --host=10.1.13.59 --port=29500 --config=config.py
