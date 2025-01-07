#!/bin/bash
#SBATCH --job-name=fistaJax100
#SBATCH --output=logs/fistaJax100.out
#SBATCH --error=logs/fistaJax100.err
#SBATCH --partition=lastgen,newgen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --share
#SBATCH --mem-per-cpu=5000
#SBATCH --exclude=node26,node27
#SBATCH --mail-user=constantin.tenzer@ens.psl.eu
#SBATCH --mail-type=ALL

# Slurm script to launch the fista_use_paper.py script

cd /home/cvaillanttenzer
source ~/.bashrc

module load anaconda

conda activate fista

#cd /home/cvaillanttenzer/Documents/fista_sofast #Remove this line if you are not using the fista_sofast repository 
#or replace with the relevant path.

python fista_use_paper_jax.py

conda deactivate
