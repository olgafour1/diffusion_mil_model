#!/bin/bash
#SBATCH --job-name=datatransfer_test
#SBATCH --output=/home/ofourkioti/Projects/CHARM/datatransfer_test.txt
#SBATCH --error=/home/ofourkioti/Projects/CHARM/datatransfer_test.err
#SBATCH --partition=data-transfer
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

#srun rsync -avP   /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/ovarian_cancer/patches/* /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/ovarian_cancer/patches/

#srun rsync -avP  /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/camelyon_data/tf_feats_256/* /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/camelyon_data/tf_feats_256/
#srun rsync -avP    /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/Nature-2019-patches/* /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon_data/Nature-2019-patches/
#srun rsync -avP   /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/lipos_flat/* /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/lipos/f lat/

srun rsync -avP  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/tf_feats_256/*  /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/multi_magnification_project/SAR_data/tf_feats_256/