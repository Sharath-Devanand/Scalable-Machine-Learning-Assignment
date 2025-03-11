#!/bin/bash
#BATCH --job-name=acp23sd-Q1_code # Replace JOB_NAME with a name you like
#SBATCH --account=default   
#SBATCH --time=02:30:00  # Change this to a longer time if you need more time
#SBATCH --nodes=1  # Specify a number of nodes
#SBATCH --mem=4G  # Request 5 gigabytes of real memory (mem)
#SBATCH --output=q1_output.txt

# Load necessary modules
module load Java/17.0.4
module load Anaconda3/2022.05

source activate myspark


# Run your Python script in parallel
spark-submit Q1_code.py
