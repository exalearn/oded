. /ccs/home/litan/miniconda3/etc/profile.d/conda.sh
conda activate mocu
export MPLCONFIGDIR=/gpfs/alpine/csc299/scratch/litan/MOCU/temp

# Feel free to use my credentials for RabbitMQ and MongoDB databases, but using your own is recommended
export RMQ_HOSTNAME=129.114.17.185
export RMQ_PORT=5672
export RMQ_USERNAME=litan
export RMQ_PASSWORD=sccDg7PxE3UjhA5L
export RADICAL_PILOT_DBURL="mongodb://litan:sccDg7PxE3UjhA5L@129.114.17.185:27017/rct-test"