pointer_layer=-2
MAX_TOKEN=4096
lr=0.0005
TEXT=news
BREAK_MODE=eos
TOKEN_PER_SAMPLE=512
BATCH_SIZE=16
RESULT=checkpoints/$TEXT/dpgpt2_eos_fs_rdrop
DEP_MODEL_PATH=/home/yangzhixian/DependencyGuided/DPLM-Transformer/checkpoints/$TEXT/gpt2_dep_decoder_eos_fs
CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 fairseq-train /home/yangzhixian/DependencyGuided/data/$TEXT/data-bin \
    --sample-break-mode $BREAK_MODE \
    --max-tokens $MAX_TOKEN \
    --save-dir $RESULT \
    --alignment-layer "$pointer_layer" \
    --dependency-model-path $DEP_MODEL_PATH \
    --dependency-model-filename checkpoint_best.pt \
    --data-path-for-load-model /home/yangzhixian/DependencyGuided/data/$TEXT/data-bin \
    --task rdrop_lm \
    --criterion reg_cross_entropy \
    --reg-alpha 1.0 \
    --arch dependency_pointer_only_transformer_lm_gpt \
    --share-decoder-input-output-embed \
    --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr "$lr" --lr-scheduler inverse_sqrt \
    --keep-last-epochs 3 \
    --distributed-world-size 6 \
    --ddp-backend legacy_ddp \
    --max-epoch 30 \
    --force-generation 0

    # --freeze-dependency-decoder
    # --tokens-per-sample $TOKEN_PER_SAMPLE \
    # --batch-size "$BATCH_SIZE" \
    # --update-freq "$update_freq" \
    #--warmup-updates "$warmup_updates" --warmup-init-lr 1e-07 \

# pointer_layer=-2
# MAX_TOKEN=4096
# lr=0.0005
# TEXT=news
# BREAK_MODE=eos
# TOKEN_PER_SAMPLE=512
# BATCH_SIZE=16
# RESULT=checkpoints/$TEXT/dpgpt2_eos_fs
# DEP_MODEL_PATH=/home/yangzhixian/DependencyGuided/DPLM-Transformer/checkpoints/$TEXT/gpt2_dependency_lm_CE_eos
# CUDA_VISIBLE_DEVICES=4,5,6,7 fairseq-train /home/yangzhixian/DependencyGuided/data/$TEXT/data-bin \
#     --task language_modeling \
#     --sample-break-mode $BREAK_MODE \
#     --max-tokens $MAX_TOKEN \
#     --save-dir $RESULT \
#     --alignment-layer "$pointer_layer" \
#     --dependency-model-path $DEP_MODEL_PATH \
#     --dependency-model-filename checkpoint_best.pt \
#     --data-path-for-load-model /home/yangzhixian/DependencyGuided/data/$TEXT/data-bin \
#     --arch dependency_pointer_transformer_lm_gpt \
#     --share-decoder-input-output-embed \
#     --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
#     --lr "$lr" --lr-scheduler inverse_sqrt \
#     --keep-last-epochs 3 \
#     --distributed-world-size 4 \
#     --ddp-backend legacy_ddp \
#     --max-epoch 30 \
#     --force-generation 0