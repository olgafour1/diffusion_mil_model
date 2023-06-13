#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=/home/olgafour1/diffusion_mil_model/camelyon_results/test_run.txt
#SBATCH --error=/home/olgafour1/diffusion_mil_model/camelyon_results/error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh
conda activate card
cd /home/olgafour1/diffusion_mil_model/

python main.py --ni --eval_best  --add_ce_loss  --exp run_test  --doc diffusion_model  --config configs/camelyon.yml --loss card_onehot_conditional --csv_file camelyon_csv_files/splits_0.csv


#for i in {0..4};
#do python run.py --experiment_name eta_1_8  --feature_path /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/camelyon_data/tf_feats_256/resnet_feats/h5_files --label_file label_files/camelyon_data.csv --csv_file camelyon_csv_files/splits_${i}.csv --lambda1 1 --epoch 200 --eta 1.8 --topk 10;
#done


