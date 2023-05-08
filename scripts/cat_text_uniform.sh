cd ..
python3 main.py \
    --exp cat_text_uniform \
    --proc_type cat \
    --dataset text8 \
    \
    --q_method uniform \
    --sched_method cosine \
    --epochs 350 \
    \
    --p_sparse True \
    --T 1000 \
    --k 27 \
