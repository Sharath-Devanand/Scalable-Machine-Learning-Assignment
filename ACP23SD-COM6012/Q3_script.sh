#!/bin/bash
#SBATCH --job-name=q3_code
#SBATCH --cpus-per-task=10 # Adjust according to your memory requirements
#SBATCH --mem-per-cpu=30G
#SBATCH --output=q3_output.txt
#SBATCH --mail-user=sdevanand1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
# Load Java module
module load Java/17.0.4

# Load Anaconda module
module load Anaconda3/2022.05

# Activate your Anaconda environment
source activate myspark

# Run your Spark script
spark-submit --driver-memory 20g --executor-memory 20g Q3_code.py
