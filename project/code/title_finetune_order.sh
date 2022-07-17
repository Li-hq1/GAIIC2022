PRETRAIN_SEED=0
GPUS='0'
PRETRAIN_SAVE_DIR='temp/tmp_data/lhq_output/title_pretrain'

# ------------order------------ #
for SEED in 0 1 2 3 4
do
    echo "seed is: ${SEED}"
    python project/code/title_finetune_order.py \
        --gpus ${GPUS} \
        --seed ${SEED} \
        --pretrain_seed ${PRETRAIN_SEED} \
        --pretrain_save_dir ${PRETRAIN_SAVE_DIR}
done
