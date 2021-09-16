Model=dp_transformer_fg0
TEXT=rocstories

for P in 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    echo "Using model : ${Model} to sample on topp ${P}..."
    CUDA_VISIBLE_DEVICES=2 fairseq-interactive ../data/$TEXT/data-bin \
        --task language_modeling \
        --batch-size 500 \
        --buffer-size 500 \
        --skip-invalid-size-inputs-valid-test \
        --input ../data/$TEXT/test_input.txt \
        --path checkpoints/$TEXT/$Model/checkpoint_best.pt \
        --sampling --beam 1 --sampling-topp $P > checkpoints/$TEXT/$Model/samples/gen_$TEXT''_$Model''_topp$P"".out
done

TEXT=news
Model=dp_transformer_fg0

for P in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
    echo "Using model : ${Model} to sample ..."
    CUDA_VISIBLE_DEVICES=4 fairseq-interactive ../data/$TEXT/data-bin \
        --task language_modeling \
        --batch-size 2000 \
        --buffer-size 2000 \
        --skip-invalid-size-inputs-valid-test \
        --path checkpoints/$TEXT/$Model/checkpoint_best.pt \
        --input ../data/$TEXT/generate_input.txt \
        --sampling --beam 1 --sampling-topp 1 > checkpoints/$TEXT/samples/gen_$TEXT''_$Model''_topp$P"".out
done