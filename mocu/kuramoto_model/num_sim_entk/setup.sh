. /ccs/home/litan/miniconda3/etc/profile.d/conda.sh
conda activate mocu
module load cuda

#export RMQ_HOSTNAME=129.114.17.185
#export RMQ_PORT=5672
#export RMQ_USERNAME=litan
#export RMQ_PASSWORD=sccDg7PxE3UjhA5L
#export RADICAL_PILOT_DBURL="mongodb://litan:sccDg7PxE3UjhA5L@129.114.17.185:27017/rct-test"

export RMQ_PASSWORD=password
export RMQ_USERNAME=admin
export RMQ_HOSTNAME=apps.marble.ccs.ornl.gov
export RMQ_PORT=31848
export RADICAL_PILOT_DBURL=mongodb://rct:rct_test@apps.marble.ccs.ornl.gov:32020/rct_test
