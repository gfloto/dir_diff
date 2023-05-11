cd ..
python3 main.py \
    --exp cat_cifar_gauss_mutual_lambda \
    --proc_type cat \
    --dataset cifar10 \
    --k 12 \
    \
    --q_method gauss \
    --sched_method linear \
    --lmbda 0.001 \
    --epochs 3000 \
    \
    --p_sparse True \
