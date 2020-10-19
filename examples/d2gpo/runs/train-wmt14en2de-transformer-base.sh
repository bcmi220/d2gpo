EXP_NAME=d2gpo
UPDATES=200000
NGPUS=8
EXP_ID=wmt14_en2de_transformer_base_up${UPDATES}_ngpus${NGPUS}_d2gpo_gaussian_std1_off0_sw0_postsoftmax_a0.1_t2.0
DATA_PATH=./data-bin/d2gpo/wmt14en2de
SAVE_PATH=./checkpoints/$EXP_NAME/$EXP_ID
LOG_PATH=./logs/$EXP_NAME/$EXP_ID.log

if [ ! -d ./checkpoints/$EXP_NAME/ ];
then 
    mkdir -p ./checkpoints/$EXP_NAME/
fi

if [ ! -d ./logs/$EXP_NAME/ ];
then 
    mkdir -p ./logs/$EXP_NAME/
fi

#CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
python fairseq_cli/train.py $DATA_PATH \
    --arch transformer_wmt_en_de \
    --max-update $UPDATES \
    --share-all-embeddings \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 --weight-decay 0.0 \
    --label-smoothing 0.1 \
    --criterion d2gpo_label_smoothed_cross_entropy \
    --d2gpo-vocab-path $DATA_PATH/d2gpo-dict.txt \
    --d2gpo-weight-path $DATA_PATH/d2gpo.gaussian_std1_off0_sw0_postsoftmax.h5 \
    --d2gpo-post-softmax \
    --d2gpo-alpha 0.1 --d2gpo-temperature 2.0 \
    --max-tokens 4096 \
    --save-dir $SAVE_PATH \
    --update-freq 1 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --no-progress-bar \
    --log-format json --log-interval 50 \
    --save-interval-updates 1000 --keep-interval-updates 200 > ${LOG_PATH} 2>&1