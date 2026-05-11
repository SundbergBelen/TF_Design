#!/bin/bash
#SBATCH --partition=glab
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=28G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=10
#SBATCH --output=out_%j.log


export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=1
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/ifs/scratch/public_softwares/mambaforge/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/ifs/scratch/public_softwares/mambaforge/etc/profile.d/conda.sh" ]; then
        . "/ifs/scratch/public_softwares/mambaforge/etc/profile.d/conda.sh"
    else
        export PATH="/ifs/scratch/public_softwares/mambaforge/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/ifs/scratch/public_softwares/mambaforge/etc/profile.d/mamba.sh" ]; then
    . "/ifs/scratch/public_softwares/mambaforge/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<
conda activate newpyrosetta


# === Print active environment ===
echo "Running in environment: $(conda info --envs | grep '*' | awk '{print $1}')"

SCRIPT="/ifs/scratch/home/bs3281/TF_Project_BS/PIPELINE_BS/Design_and_Score_backrub.py"

# run design script 
python -u $SCRIPT --config "$1"