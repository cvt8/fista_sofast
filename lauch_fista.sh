#!/bin/bash
#SBATCH --job-name=fista
#SBATCH --output=logs/fista.out
#SBATCH --error=logs/fista.err
#SBATCH --partition=gnt,lastgen, newgen
#SBATCH --array=1-1:1
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

cd /home/cvaillanttenzer/Documents/fista_sofast

python fista_use_paper.py

conda deactivate

#fastgen
