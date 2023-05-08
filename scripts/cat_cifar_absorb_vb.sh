cd ..
python3 main.py \
    --exp cat_cifar_absorb_mutual_vb \
    --proc_type cat \
    --dataset cifar10 \
    --k 12 \
    \
    --q_method absorbing \
    --sched_method mutual_info \
    --epochs 3000 \
