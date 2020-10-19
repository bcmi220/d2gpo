N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epoch

# fastText
FASTTEXT=/path/to/fastText/build/fasttext 

NAME=wmt14en2de.fasttext.300d
INPUT=./examples/d2gpo/wmt14_en_de/all.en-de
OUTPUT=./examples/d2gpo/wmt14_en_de/$NAME
LOG=./examples/d2gpo/wmt14_en_de/${NAME}.log

$FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 300 -thread $N_THREADS -ws 5 -neg 10 -input $INPUT -output $OUTPUT 1>$LOG 2>&1