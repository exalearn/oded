A Workflow-based MOCU Implementation using RADICAL Cybertools (RCT EnTK)

**Installation (MOCU and RCT)**

```
. /ccs/home/litan/miniconda3/etc/profile.d/conda.sh
conda create -n mocu python=3.6 # python 3.7 won't work
conda activate mocu
pip install radical.entk radical.pilot radical.saga radical.utils --upgrade

conda install scipy
pip install matplotlib
conda install -c engility pytorch
make
```

**Run**

```
source setup.sh
```

Next see `run.sh` for possible commands to run on ORNL Summit.

**Performance**

See the `profiling` directory for scaling test results.
