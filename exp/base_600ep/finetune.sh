work_path=$(dirname $0)
filename=$(basename $work_path)
partition=$1
gpus=$2
datapath=$3
OMP_NUM_THREADS=1 \
srun -p ${partition} -n ${gpus}  --ntasks-per-node=8 --cpus-per-task=12 --gres=gpu:8 \
python -u main_finetune.py \
    --batch_size 128 \
    --model mixmim_base \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.7 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --data_path ${datapath} \
    --output_dir ${work_path}/ft_ckpt \
    --log_dir ${work_path}/ft_ckpt \
    --resume ${work_path}/ft_ckpt/checkpoint.pth \
    --port 29528 \
    --finetune ${work_path}/ckpt/checkpoint.pth
