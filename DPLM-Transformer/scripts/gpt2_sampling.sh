Model=dpgpt2_eos
TEXT=news
for P in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
    echo "Using model : ${Model} to sample ..."
    CUDA_VISIBLE_DEVICES=3 fairseq-interactive ../data/news/gpt2-data-bin \
        --task language_modeling \
        --batch-size 400 \
        --buffer-size 400 \
        --skip-invalid-size-inputs-valid-test \
        --path checkpoints/news/$Model/checkpoint_best.pt \
        --input ../data/news/generate_input.txt \
        --sampling --beam 1 --sampling-topp $P > checkpoints/$TEXT/samples/gen_$TEXT''_$Model''_topp$P.out
done


Model=dpgpt2_eos
TEXT=rocstories
for P in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
do
    echo "Using model : ${Model} to sample on topp ${P}..."
    CUDA_VISIBLE_DEVICES=0 fairseq-interactive ../data/$TEXT/gpt2-data-bin \
        --task language_modeling \
        --batch-size 100 \
        --buffer-size 100 \
        --skip-invalid-size-inputs-valid-test \
        --path checkpoints/$TEXT/$Model/checkpoint_best.pt \
        --input ../data/$TEXT/test_input.bpe \
        --sampling --beam 1 --sampling-topp $P > checkpoints/$TEXT/$Model/samples/gen_$TEXT''_$Model''_topp$P"".out
done

