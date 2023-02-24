# PRETRAIN_SEED=0
# GPUS='0'
# PRETRAIN_SAVE_DIR='temp/tmp_data/lhq_output/dataset_scale/13w'
# SAVE_DIR='temp/tmp_data/lhq_output/finetune_scale/6k/'

# # ------------order------------ #
# for SEED in 0
# do
#     echo "seed is: ${SEED}"
#     python project/code/title_finetune_ablation.py \
#         --gpus ${GPUS} \
#         --seed ${SEED} \
#         --pretrain_seed ${PRETRAIN_SEED} \
#         --pretrain_save_dir ${PRETRAIN_SAVE_DIR} \
#         --save_dir ${SAVE_DIR}
# done



PRETRAIN_SEED=0
SEED=4
GPUS='2'
PRETRAIN_SAVE_DIR='temp/tmp_data/lhq_output/dataset_scale/13w'


# ------------order------------ #
for SCALE in 9000 8000 7000 6000 5000 4000 3000 2000 1000 500 200 100
do
    SAVE_DIR="temp/tmp_data/lhq_output/finetune_scale/"$SCALE"/"
    echo "seed is: ${SEED}"
    python project/code/title_finetune_ablation_scale.py \
        --gpus ${GPUS} \
        --seed ${SEED} \
        --pretrain_seed ${PRETRAIN_SEED} \
        --pretrain_save_dir ${PRETRAIN_SAVE_DIR} \
        --save_dir ${SAVE_DIR} \
        --scale ${SCALE}
done