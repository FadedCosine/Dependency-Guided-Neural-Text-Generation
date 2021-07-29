Model=dpt_fg_0_rdrop
TEXT=rocstories
P=0.3
for Model in dpt_fg_0_rdrop transformer_lm_eos
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
P=0.4
for Model in dpt_fg_0_rdrop transformer_lm_eos
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

K=3
for Model in dpt_fg_0_rdrop transformer_lm_eos
do
    echo "Using model : ${Model} to sample on topk ${K}..."
    CUDA_VISIBLE_DEVICES=3 fairseq-interactive ../data/$TEXT/data-bin \
        --task language_modeling \
        --batch-size 500 \
        --buffer-size 500 \
        --skip-invalid-size-inputs-valid-test \
        --path checkpoints/$TEXT/$Model/checkpoint_best.pt \
        --input ../data/$TEXT/test_input.txt \
        --sampling --beam 1 --sampling-topk $K > checkpoints/$TEXT/$Model/samples/gen_$TEXT''_$Model''_topk$K"".out
done
# for P in 0.5 0.6 0.7 0.8 0.9 0.95
# do
#     echo "Using model : ${Model} to sample on topp ${P}..."
#     CUDA_VISIBLE_DEVICES=2 fairseq-interactive ../data/$TEXT/data-bin \
#         --task language_modeling \
#         --batch-size 500 \
#         --buffer-size 500 \
#         --skip-invalid-size-inputs-valid-test \
#         --input ../data/$TEXT/test_input.txt \
#         --path checkpoints/$TEXT/$Model/checkpoint_best.pt \
#         --sampling --beam 1 --sampling-topp $P > checkpoints/$TEXT/$Model/samples/gen_$TEXT''_$Model''_topp$P"".out
# done
        
# for K in 1 5 10 30 50 70 100
# do
#     echo "Using model : ${Model} to sample on topk ${K}..."
#     CUDA_VISIBLE_DEVICES=3 fairseq-interactive ../data/$TEXT/data-bin \
#         --task language_modeling \
#         --batch-size 500 \
#         --buffer-size 500 \
#         --skip-invalid-size-inputs-valid-test \
#         --path checkpoints/$TEXT/$Model/checkpoint_best.pt \
#         --input ../data/$TEXT/test_input.txt \
#         --sampling --beam 1 --sampling-topk $K > checkpoints/$TEXT/$Model/samples/gen_$TEXT''_$Model''_topk$K"".out
# done

# CUDA_VISIBLE_DEVICES=4 fairseq-interactive ../data/$TEXT/data-bin \
#     --task language_modeling \
#     --batch-size 100 \
#     --buffer-size 100 \
#     --skip-invalid-size-inputs-valid-test \
#     --path checkpoints/$TEXT/$Model/checkpoint_best.pt \
#     --input ../data/$TEXT/generate_input.txt \
#     --sampling --beam 1 --sampling-topk 10 > gen_debug.out