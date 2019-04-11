MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

mpirun \
-npernode 4 \
-np 8 \
-x PATH \
-x LIBRARY_PATH \
-x LD_LIBRARY_PATH \
-x CUDA_HOME \
-x CUDA_CACHE_DISABLE=1 \
python main.py \
--dist_init_method $MASTER_ADDR \
--config lenet_distributed_kfac.json
