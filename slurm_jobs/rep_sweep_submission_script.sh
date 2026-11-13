#!/bin/bash


nl_str=(0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
# kernel_init=.2

for nls in ${nl_str[@]}; do
    sbatch rep_sweep.sbatch --dm_input_mixing $nls # --kernel_init_std $kernel_init
done
