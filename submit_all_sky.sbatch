#!/bin/bash
#SBATCH --job-name=all_sky
#SBATCH --output=example_sbatch.out
#SBATCH --error=example_sbatch.err
#SBATCH --time=24:00:00
#SBATCH --partition=kicpaa
#SBATCH --account=kicpaa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20

echo "HELLO!"

module load python

#jupyter nbconvert --to script analyze_all_sky_map_km.ipynb
python3 analyze_all_sky_map_km.py