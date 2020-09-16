#!/bin/bash

# mocu_old: n - number of experiments, i - index of current experiment, p - psi
python new/mocu/scripts/example_scaling.py -n 3 -p 40
python mocu_entk.py -n 1 -p 40
python mocu_entk.py -n 10 -p 50

# mocu_new (current MOCU): n - number of OED runs, t - value of Theta, p - value of Psi, s - value of S
python mocu_entk_new.py -n 128 -t 16 -p 16 -s 16
python mocu_entk_new.py -n 128 -t 64 -p 16 -s 16
python mocu_entk_new.py -n 128 -t 16 -p 64 -s 16
python mocu_entk_new.py -n 128 -t 16 -p 16 -s 64
python mocu_entk_new.py -n 128 -t 64 -p 64 -s 64 (302.1s vs 4788.0s using one core)
