TEXT=rocstories
CUDA_VISIBLE_DEVICES=7 python run_rnn_baseline.py --train_batch_size 120 \
    --data ../data/$TEXT \
    --dataname $TEXT \
    --dependency_path ../data/$TEXT/dependency \
    --model LSTM \
    --dropouti 0.4 \
    --dropouth 0.25 \
    --cuda \
    --seed 141 \
    --lr 30 \
    --epochs 500 \
    --trained_epoches 0 \
    --save checkpoints/$TEXT/AWDLSTM_$TEXT.pt