Model=dpgpt2_eos
TEXT=news
Path=checkpoints/$TEXT/samples
for P in 1
do
    echo "Decode ${Model} topp ${P} files..."
    python util/multiprocessing_bpe_decode.py \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs $Path/$Model''_topp$P''.bpe \
        --outputs $Path/$Model''_topp$P''.txt \
        --keep-empty \
        --workers 60; \
done