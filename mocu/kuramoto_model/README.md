A Workflow-based MOCU Implementation using RADICAL Cybertools

**Installation (MOCU and RCT)**

```
. /ccs/home/litan/miniconda3/etc/profile.d/conda.sh
conda create -n mocu python=3.6 # python 3.7 won't work
conda activate mocu
pip install radical.entk radical.pilot radical.saga radical.utils --upgrade

pip install pycuda
```

**Two Levels of Parallelism**

1. Parallelizing different simulations (one simulation per GPU) using RADICAL EnTK

* See `num_sim_entk` folder.

2. Parallelizing computation of expected remaining MOCU (one MOCU function call per GPU) using RADICAL RP (function execution feature)

* See `remainingMOCU_func_exec_rp` folder.
