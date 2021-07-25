A Workflow-based MOCU Implementation using RADICAL Cybertools (RCT EnTK)

**Installation (MOCU and RCT)**

```
. /ccs/home/litan/miniconda3/etc/profile.d/conda.sh
conda create -n mocu python=3.6 # python 3.7 won't work
conda activate mocu
pip install radical.entk radical.pilot radical.saga radical.utils --upgrade

pip install pycuda
```

**Two Types of Parallelism**

```
source setup.sh
python mocu_entk.py -n 120 # -n specifies the number of simulations and each simulation runs on one GPU
```

<!--Next see `run.sh` for possible commands to run on ORNL Summit.

**Performance**

See the `profiling` directory for scaling test results.-->
