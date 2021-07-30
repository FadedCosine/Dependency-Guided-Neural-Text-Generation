TEXT=news
# CONTEXT_LEN=36
# CUDA_VISIBLE_DEVICES=4 python train_dp.py --train_batch_size 200 \
#     --data ../data/$TEXT \
#     --dataname $TEXT \
#     --dependency_path ../data/$TEXT/dependency \
#     --dropout 0.45 \
#     --dropouth 0.3 \
#     --dropouti 0.5 \
#     --wdrop 0.45 \
#     --chunk_size 10 \
#     --cuda \
#     --seed 141 \
#     --lr 30 \
#     --context_length $CONTEXT_LEN \
#     --save checkpoints/$TEXT/dp_ON_$TEXT''_context_36.pt \
#     --do_eval
CONTEXT_LEN=-1
CUDA_VISIBLE_DEVICES=4 python train_dp.py --train_batch_size 200 \
    --data ../data/$TEXT \
    --dataname $TEXT \
    --dependency_path ../data/$TEXT/dependency \
    --dropout 0.45 \
    --dropouth 0.3 \
    --dropouti 0.5 \
    --wdrop 0.45 \
    --chunk_size 10 \
    --cuda \
    --seed 141 \
    --lr 30 \
    --context_length $CONTEXT_LEN \
    --save checkpoints/$TEXT/dp_ON_$TEXT''_all_context.pt \
    --do_eval