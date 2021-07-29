TEXT=rocstories
# for SPLIT in train valid test; do \
#     python -m examples.roberta.multiprocessing_bpe_encoder \
#         --encoder-json gpt2_bpe/encoder.json \
#         --vocab-bpe gpt2_bpe/vocab.bpe \
#         --inputs /home/yangzhixian/DependencyGuided/data/$TEXT/${SPLIT}.txt \
#         --outputs /home/yangzhixian/DependencyGuided/data/$TEXT/${SPLIT}.bpe \
#         --keep-empty \
#         --workers 60; \
# done
python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json gpt2_bpe/encoder.json \
    --vocab-bpe gpt2_bpe/vocab.bpe \
    --inputs /home/yangzhixian/DependencyGuided/data/$TEXT/test_input.txt \
    --outputs /home/yangzhixian/DependencyGuided/data/$TEXT/test_input.bpe \
    --keep-empty \
    --workers 60