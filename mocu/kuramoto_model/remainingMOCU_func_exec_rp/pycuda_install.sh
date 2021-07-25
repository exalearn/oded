module load gcc/7.4.0
find /ccs/home/litan/.cache -type f -name 'pycuda*' -delete
export PYCUDA_CACHE_DIR=/gpfs/alpine/csc299/scratch/litan/MOCU/Byung-Jun/new/ExaLearn-ODED-Kuramoto-main/N7/pycuda_cache/pycuda/compiler-cache-v1
pip install --cache-dir /gpfs/alpine/csc299/scratch/litan/MOCU/Byung-Jun/new/ExaLearn-ODED-Kuramoto-main/N7/pycuda_cache/pycuda/compiler-cache-v1 pycuda==2020.1
