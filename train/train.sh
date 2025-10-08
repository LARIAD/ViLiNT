#!/bin/bash
#SBATCH --job-name=TrainLint        # nom du job
#SBATCH --output=TrainLint%j.out    # fichier de sortie (%j = job ID)
#SBATCH --error=TrainLint%j.err     # fichier d’erreur (%j = job ID)
#SBATCH --constraint=h100           # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1                   # reserver 1 nœud
#SBATCH --ntasks=1                  # reserver 1 tache (ou processus)
#SBATCH --gres=gpu:4                # reserver 4 GPU
#SBATCH --cpus-per-task=24          # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=20:00:00             # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread        # desactiver l’hyperthreading
#SBATCH --account=rka@h100          # comptabilite V100
module purge                        # nettoyer les modules herites par defaut
module load arch/h100               # charger les modules
module load cuda/12.8.0
module load miniforge/24.11.3
conda activate $HOME/vilint
set -x              # activer l’echo des commandes
srun python train.py -c config/vilint.yaml                  
# executer son script