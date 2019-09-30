#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=100:00:00
#SBATCH -p all
#SBATCH --mem-per-cpu=1000
#SBATCH --output=slurm-XX.out
#SBATCH --error=slurm-XX.err
#SBATCH --mail-user=ting.yuansen.astro@gmail.com
#SBATCH --mail-type=ALL

python download_spectra.py XX
