. /ccs/home/litan/miniconda3/etc/profile.d/conda.sh
conda activate mocu
module load cuda
#module load cuda/11.1.1
#module load gcc/8.1.1

##ulimit -n unlimited
##ulimit -n 2048

export RMQ_HOSTNAME=129.114.17.185
export RMQ_PORT=5672
export RMQ_USERNAME=<username>
export RMQ_PASSWORD=<password>
export RADICAL_PILOT_DBURL="mongodb://<username>:<password>@129.114.17.185:27017/rct-test"
