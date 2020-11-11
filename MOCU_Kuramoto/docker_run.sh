#!/bin/bash

for i in {0..2};
do
   python runMainForPerformanceMeasure.py -n 3 -i $i CUDA_VISIBLE_DEVICES=$i
done
