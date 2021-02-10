#!/bin/bash -l

#SBATCH -J mdins
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --partition=med
#SBATCH -o md_ins.out
#SBATCH -e md_ins.err
#SBATCH -t 24:00:00

module load python3
module load gromacs

srun python MD_INS.py mdins_input25K.txt
