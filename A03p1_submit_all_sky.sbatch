#!/bin/bash
#SBATCH --job-name=all_sky
#SBATCH --output=example_sbatch.out
#SBATCH --error=example_sbatch.err
#SBATCH --time=24:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10

module load python
python3 analyze_all_sky_map.py
