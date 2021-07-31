TEXT=news
CONTEXT_LEN=-1
MODEL=dp_ON_$TEXT''_context_$CONTEXT_LEN
for P in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 
do
    CUDA_VISIBLE_DEVICES=4 python generate.py  \
        --data ../data/$TEXT \
        --dataname $TEXT \
        --outf checkpoints/$TEXT/$MODEL''_samples/gen_$MODEL''_topp$P.txt \
        --topp $P \
        --seed 910 \
        --words 100000 \
        --context_length $CONTEXT_LEN \
        --is_dp_model \
        --cuda \
        --checkpoint checkpoints/$TEXT/$MODEL''.pt
done
for K in 1 3 5 7 9 10 20 30 40 50
do
    CUDA_VISIBLE_DEVICES=4 python generate.py  \
        --data ../data/$TEXT \
        --dataname $TEXT \
        --outf checkpoints/$TEXT/$MODEL''_samples/gen_$MODEL''_topk$K.txt \
        --topk $K \
        --seed 910 \
        --words 100000 \
        --context_length $CONTEXT_LEN \
        --is_dp_model \
        --cuda \
        --checkpoint checkpoints/$TEXT/$MODEL''.pt
done
# TEXT=news
# MODEL=ON_news
# CUDA_VISIBLE_DEVICES=6 python generate.py  \
#     --data ../data/$TEXT \
#     --dataname $TEXT \
#     --outf debug.txt \
#     --topk 5 \
#     --seed 141 \
#     --words 100000 \
#     --cuda \
#     --checkpoint checkpoints/$TEXT/$MODEL''.pt