TEXT=rocstories
fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref ../data/$TEXT/train.bpe \
    --validpref ../data/$TEXT/valid.bpe \
    --testpref ../data/$TEXT/test.bpe \
    --destdir ../data/$TEXT/gpt2-data-bin \
    --workers 60