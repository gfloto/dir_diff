cd ..
python3 main.py \
    --exp cat_cifar_uniform \
    --proc_type cat \
    --dataset cifar10 \
    --k 12 \
    \
    --q_method uniform \
    --sched_method cosine \
    --epochs 3000 \
    \
    --p_sparse True \