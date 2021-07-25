An RCT RP Workflow Parallelizing Computation of Remaining MOCU using the Function Execution Feature

**Installation (MOCU and RCT)**

```
. /ccs/home/litan/miniconda3/etc/profile.d/conda.sh
conda create -n mocu python=3.7
conda activate mocu
pip install radical.saga radical.utils --upgrade
pip install git+https://github.com/radical-cybertools/radical.pilot.git@feature/funcs_v2

pip install pycuda
```

**Run**

```
source setup.sh
python remainingMOCU_func_exec_rp.py
```
