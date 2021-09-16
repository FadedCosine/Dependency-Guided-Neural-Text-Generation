TEXT=rocstories
fairseq-preprocess \
    --only-source \
    --trainpref ../data/$TEXT/train.txt \
    --validpref ../data/$TEXT/valid.txt \
    --testpref ../data/$TEXT/test.txt \
    --destdir ../data/$TEXT/data-bin \
    --workers 20