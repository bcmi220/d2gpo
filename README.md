# D2GPo: Data-depedent Gaussian Prior Objective

The implementation of "Data-dependent Gaussian Prior Objective for Language Generation"

> This code is based on [Fairseq](https://github.com/pytorch/fairseq)

## Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.6
* [fastText](https://github.com/facebookresearch/fastText)
* gensim
* scikit-learn

## Prepare Training Data

```shell
# download the wmt14 en2de data
bash runs/prepare-wmt14en2de.sh

mkdir -p ./data-bin/d2gpo/wmt14en2de
TEXT=./examples/d2gpo/wmt14_en_de
DATA_PATH=./data-bin/d2gpo/wmt14en2de

# binarize the data
python fairseq_cli/preprocess.py \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --destdir $DATA_PATH \
    --thresholdtgt 0 --thresholdsrc 0 \
    --joined-dictionary \
    --workers 20

# generate the full vocabulary for D2GPo
python examples/d2gpo/scripts/generate_d2gpo_vocab.py \
    --fairseq_vocab_path $DATA_PATH/dict.de.txt \
    --d2gpo_vocab_path $DATA_PATH/d2gpo-dict.txt

```

## Pretrain embeddings with FastText

```shell

cat $TEXT/train.en $TEXT/train.de > all.en-de

./examples/d2gpo/runs/pretrain-fasttext-wmt14en2de.sh

# generate the embedding for the d2gpo vocabulary
python examples/d2gpo/scripts/generate_d2gpo_embedding.py \
    --pretrain_emb_model $TEXT/wmt14en2de.fasttext.300d.bin \
    --d2gpo_vocab_path $DATA_PATH/d2gpo-dict.txt \
    --emb_output_path $DATA_PATH/d2gpo-dict.vec \
    --w2v_emb_output_path $DATA_PATH/d2gpo-dict.w2vec

```

## Generate the gaussian prior distribution
```shell

# generate a unique order for words in vocabulary according to the embedding similarity
python examples/d2gpo/scripts/generate_d2gpo_order.py \
    --w2v_emb_path $DATA_PATH/d2gpo-dict.w2vec \
    --d2gpo_order_txt $DATA_PATH/d2gpo.order.txt \
    --d2gpo_order_idx $DATA_PATH/d2gpo.order.idx

# generate the prior distribution
python examples/d2gpo/scripts/generate_d2gpo_distribution.py \
    --d2gpo_mode gaussian \
    --d2gpo_gaussian_std 1 \
    --d2gpo_gaussian_offset 0 \
    --d2gpo_sample_width 0 \
    --d2gpo_softmax_position postsoftmax \
    --d2gpo_softmax_temperature 1.0 \
    --d2gpo_order_idx $DATA_PATH/d2gpo.order.idx \
    --d2gpo_distribution_output $DATA_PATH/d2gpo.gaussian_std1_off0_sw0_postsoftmax.h5

```

## Train the full model

```shell

# transformer base
./examples/d2gpo/runs/train-wmt14en2de-transformer-base.sh

```

## Inference on trained model


```shell
CKPT=/path/to/checkpoint_best.pt
python fairseq_cli/translate.py $DATA_PATH \
    --task translation \
    --source-lang en --target-lang de \
    --path $CKPT \
    --buffer-size 2000 --batch-size 64 --beam 4 --remove-bpe --lenpen 0.6 \
    --input $TEXT/test.en \
    --output ./result/wmt14_en2de_test.de.pred \
    --no-print --truncate-size 512

```


## Citation

```bibtex
@inproceedings{
    Li2020Data-dependent,
    title={Data-dependent Gaussian Prior Objective for Language Generation},
    author={Zuchao Li and Rui Wang and Kehai Chen and Masso Utiyama and Eiichiro Sumita and Zhuosheng Zhang and Hai Zhao},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=S1efxTVYDr}
}
```
