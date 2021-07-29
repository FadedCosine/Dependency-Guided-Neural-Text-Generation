Model=gpt2_eos
TEXT=rocstories
Path=checkpoints/$TEXT/$Model/samples
for P in 0.5 0.6 0.7 0.8 0.9 0.95
do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs $Path/$Model''_topp$P''.bpe \
        --outputs $Path/$Model''_topp$P''.txt \
        --keep-empty \
        --workers 60; \
done

for K in 1 5 10 30 50 70 100
do
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs $Path/$Model''_topk$K''.bpe \
        --outputs $Path/$Model''_topk$K''.txt \
        --keep-empty \
        --workers 60; \
done
# python -m examples.roberta.multiprocessing_bpe_encoder \
#     --encoder-json gpt2_bpe/encoder.json \
#     --vocab-bpe gpt2_bpe/vocab.bpe \
#     --inputs /home/yangzhixian/DependencyGuided/DPLM-Transformer/gpt2_gen_debug_withbpe.txt \
#     --outputs /home/yangzhixian/DependencyGuided/DPLM-Transformer/gpt2_gen_debug_withbpe_decode.txt \
#     --keep-empty \
#     --workers 60