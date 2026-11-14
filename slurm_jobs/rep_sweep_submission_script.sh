#!/bin/bash

current_datetime=`date +"%Y-%m-%d_%T"`

if [ ! -z "$@" ]; then
    current_datetime="$@"_$current_datetime
fi
out_folder=/mnt/fac/CX500139_DS1/results/modularity_sweeps/$current_datetime
echo $out_folder
mkdir $out_folder
# nl_str=(0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
nl_str=(0 5)

for nls in ${nl_str[@]}; do
    sbatch slurm_jobs/rep_sweep.sbatch --dm_input_mixing $nls --output_mainfolder $out_folder
done
