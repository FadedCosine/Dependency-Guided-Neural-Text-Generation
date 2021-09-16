Model=dpgpt2_eos
TEXT=news
Path=checkpoints/$TEXT/samples
for P in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
    echo "Decode ${Model} topp ${P} files..."
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs $Path/$Model''_topp$P''.bpe \
        --outputs $Path/$Model''_topp$P''.txt \
        --keep-empty \
        --workers 60; \
done