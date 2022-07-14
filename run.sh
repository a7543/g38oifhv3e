#!/usr/bin/bash
export DATA=coldata_dbg/dataset
#export RANK=0
#export LOCAL_RANK=0
#export WORLD_SIZE=1
#export MASTER_ADDR=127.0.0.1
#export MASTER_PORT=29500

#python go.py --config=config.py
colossalai run --nproc_per_node=1 go.py --config=config.py --from_torch --world_size=1
#colossalai run --nproc_per_node=1 go.py --config=config.py --from_torch --world_size=2 --host 10.1.13.62,10.1.13.63 --master_addr 10.1.13.63
#colossalai run --nproc_per_node=1 go.py --config=config.py --from_torch --world_size=4 --host 10.1.13.60,10.1.13.61,10.1.13.62,10.1.13.63 --master_addr 10.1.13.63
echo --hostfile ./hostfile
#python /home/nfs/zsc/yuan/ColossalAI-Examples/language/gpt/train_gpt.py --rank=0 --world_size=1 --host=10.1.13.59 --port=29500 --config=config.py
