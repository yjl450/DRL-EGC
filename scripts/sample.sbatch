#!/bin/bash

#SBATCH --job-name=pfrla3c-r3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
# #SBATCH --gres=gpu:p40:
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=yjl450@nyu.edu
#SBATCH --output=slurm_%j.out


echo "sample pfrl a3c"
echo "python scripts/pfrl_sample_a3c.py --reward 3"
module purge
module load anaconda3/2020.07
source activate
source deactivate
conda deactivate
conda activate capstone

python scripts/pfrl_sample_a3c.py --reward 3
echo "FINISH"
