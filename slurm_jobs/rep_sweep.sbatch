#!/bin/bash
#SBATCH --job-name=rep_sweep_%a_%A                                           
#SBATCH --output=.logs/rep_sweep_%a_%A.out                      
#SBATCH --error=.logs/rep_sweep_%a_%A.err                            
#SBATCH --cpus-per-task=4
#SBATCH --mem 15gb
#SBATCH --array=1
#SBATCH --time=11:59:59

# NL_STRENGTH SHOULD BE EXTERNALLY PROVIDED
part=$SLURM_ARRAY_TASK_ID
arr=$SLURM_ARRAY_JOB_ID

nl_denom=100

inp=6
rep_dim=500
model_epochs=20
group_size=3
n_groups=2
group_width=100
activity_reg=0
group_overlap=$group_size
n_reps=30

time rep_mod --n_reps $n_reps --output_subfolder rep_sweep_$part\_nls$nl_strength --group_size $group_size --n_groups $n_groups --model_epochs $model_epochs --group_overlap $group_overlap --activity_reg $activity_reg --group_width $group_width --tasks_per_group $part --input_dim $inp --rep_dim $rep_dim --dm_input_mixing_denom $nl_denom "$@"
