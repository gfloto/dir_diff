cd ..
python3 ../main.py \
    --exp cat_text_absorb \
    --proc_type cat \
    --dataset text8 \
    \
    --q_method absorbing \
    --sched_method mutual_info \
    --lmbda 0.01
    --epochs 350 \
    \
    --p_sparse True \
    --T 1000 \
    --k 27 \

