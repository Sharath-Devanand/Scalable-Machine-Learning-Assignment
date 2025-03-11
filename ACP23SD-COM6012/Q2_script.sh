#!/bin/bash
#BATCH --job-name=q2_code
#SBATCH --cpus-per-task=5 # Adjust according to your memory requirements
#SBATCH --mem-per-cpu=10G
#SBATCH --output=Output/q2_output.txt
#SBATCH --error=error_q2.txt
# Load Java module
module load Java/17.0.4

# Load Anaconda module
module load Anaconda3/2022.05

# Activate your Anaconda environment
source activate myspark

# Run your Spark script
spark-submit --driver-memory 10g --executor-memory 10g Q2_code.py
