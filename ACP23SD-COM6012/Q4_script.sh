#!/bin/bash
#SBATCH --job-name=q4_code
#SBATCH --cpus-per-task=10 # Adjust according to your memory requirements
#SBATCH --mem-per-cpu=20G
#SBATCH --output=q4_output.txt
#SBATCH --error=error_q4.txt
# Load Java module
module load Java/17.0.4

# Load Anaconda module
module load Anaconda3/2022.05

# Activate your Anaconda environment
source activate myspark

# Run your Spark script
spark-submit --driver-memory 20g --executor-memory 20g Q4_code.py
