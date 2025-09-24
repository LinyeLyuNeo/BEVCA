#!/usr/bin/env bash

# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
# PORT=${PORT:-29503}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox



# filepath: /data/lyulinye/bevformer/tools/dist_test.sh
#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_v2.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox