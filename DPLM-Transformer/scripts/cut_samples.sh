Model=dpt_fg_0_rdrop
TEXT=rocstories
Path=checkpoints/$TEXT/$Model/samples

for P in 0.5 0.6 0.7 0.8 0.9 0.95
do
    grep ^H $Path/gen_$TEXT''_$Model''_topp$P"".out | cut -f3- > $Path/$Model''_topp$P''.out
done

for K in 1 5 10 30 50 70 100
do
    grep ^H $Path/gen_$TEXT''_$Model''_topk$K"".out | cut -f3- > $Path/$Model''_topk$K''.out
done