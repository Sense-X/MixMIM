work_path=$(dirname $0)
filename=$(basename $work_path)
partition=$1
gpus=$2
datapath=$3
OMP_NUM_THREADS=1 \
srun -p ${partition} -n ${gpus} --ntasks-per-node=8 --cpus-per-task=16 --gres=gpu:8 \
python -u main_pretrain.py \
    --batch_size 128 \
    --model mixmim_base \
    --norm_pix_loss \
    --mask_ratio 0.5 \
    --epochs 600 \
    --accum_iter 1 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${datapath} \
    --output_dir ${work_path}/ckpt \
    --log_dir ${work_path}/log \
    --resume ${work_path}/ckpt/checkpoint.pth
