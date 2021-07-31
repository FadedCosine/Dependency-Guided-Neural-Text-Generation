TEXT=rocstories
MODEL=ON_$TEXT
for P in 0.3 0.4 # 0.7 0.8 0.9 0.95
do
    CUDA_VISIBLE_DEVICES=0 python generate.py  \
        --data ../data/$TEXT \
        --dataname $TEXT \
        --task story_gen \
        --outf checkpoints/$TEXT/$MODEL''_eval_samples/gen_$MODEL''_topp$P.txt \
        --topp $P \
        --seed 141 \
        --cuda \
        --checkpoint checkpoints/$TEXT/$MODEL''.pt
done
# for K in 1 5 10 30 50 70 100
# do
#     CUDA_VISIBLE_DEVICES=0 python generate.py  \
#         --data ../data/$TEXT \
#         --dataname $TEXT \
#         --task story_gen \
#         --outf checkpoints/$TEXT/$MODEL''_samples/gen_$MODEL''_topk$K.txt \
#         --topk $K \
#         --seed 141 \
#         --cuda \
#         --checkpoint checkpoints/$TEXT/$MODEL''.pt
# done
