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
which python3
which python
which srun
hostname

scriptdir="/home/tfharrel/MD/Scripts/"

srun python ${scriptdir}MD_INS_mpi_new.py ${scriptdir}mdins_input10K.txt
