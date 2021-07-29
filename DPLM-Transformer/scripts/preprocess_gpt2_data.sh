TEXT=rocstories
fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref /home/yangzhixian/DependencyGuided/data/$TEXT/train.bpe \
    --validpref /home/yangzhixian/DependencyGuided/data/$TEXT/valid.bpe \
    --testpref /home/yangzhixian/DependencyGuided/data/$TEXT/test.bpe \
    --destdir /home/yangzhixian/DependencyGuided/data/$TEXT/gpt2-data-bin \
    --workers 60