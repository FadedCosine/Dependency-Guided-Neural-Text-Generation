TEXT=news
MAX_TOKEN=2048
BREAK_MODE=eos
WARMUP_UPDATES=0
LR=5e-5
RESULT=checkpoints/$TEXT/gpt2_dep_decoder_$BREAK_MODE
# CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 \
#     /home/yangzhixian/fairseq/fairseq_cli/train.py \
#     --task dependency_modeling /home/yangzhixian/DependencyGuided/data/$TEXT/gpt2-data-bin \
#     --save-dir $RESULT \
#     --dependency /home/yangzhixian/DependencyGuided/data/$TEXT/dependency --dependency-suffix .head \
#     --arch hf_dpgpt2 --pretrained-model /home/yangzhixian/pretrained_model/gpt2-base \
#     --criterion dependency_cross_entropy \
#     --sample-break-mode $BREAK_MODE --max-tokens $MAX_TOKEN \
#     --save-dir $RESULT \
#     --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-8  \
#     --weight-decay 0.00 --clip-norm 0.0 \
#     --lr $LR --warmup-updates $WARMUP_UPDATES  \
#     --keep-last-epochs 3 \
#     --distributed-world-size 6 \
#     --ddp-backend legacy_ddp \
#     --is-train-dependency \
#     --max-epoch 50
MAX_TOKEN=2048
RESTORE=checkpoints/$TEXT/gpt2_dep_decoder_$BREAK_MODE
RESULT=checkpoints/$TEXT/dpgpt2_$BREAK_MODE
CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 fairseq-train /home/yangzhixian/DependencyGuided/data/$TEXT/gpt2-data-bin \
    --task language_modeling \
    --save-dir $RESULT \
    --restore-file $RESTORE/checkpoint_best.pt \
    --reset-dataloader \
    --reset-lr-scheduler \
    --reset-meters \
    --reset-optimizer \
    --arch hf_dpgpt2 --pretrained-model /home/yangzhixian/pretrained_model/gpt2-base \
    --sample-break-mode $BREAK_MODE --max-tokens $MAX_TOKEN \
    --save-dir $RESULT \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-8  \
    --weight-decay 0.00 --clip-norm 0.0 \
    --lr $LR --warmup-updates $WARMUP_UPDATES  \
    --keep-last-epochs 5 \
    --distributed-world-size 6 \
    --ddp-backend legacy_ddp \
    --max-epoch 42
    