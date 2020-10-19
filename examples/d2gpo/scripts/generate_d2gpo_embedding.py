import sys
import argparse
from gensim.models.wrappers import FastText
from gensim.scripts.glove2word2vec import glove2word2vec


if __name__ == '__main__':
    '''
    '''

    parser = argparse.ArgumentParser(description='D2GPo embedding preparation')
    parser.add_argument("--pretrain_emb_model", type=str, default="")
    parser.add_argument("--d2gpo_vocab_path", type=str, default="")
    parser.add_argument("--emb_output_path", type=str, default="")
    parser.add_argument("--w2v_emb_output_path", type=str, default="")
    args = parser.parse_args()

    model = FastText.load_fasttext_format(args.pretrain_emb_model)

    with open(args.d2gpo_vocab_path, 'r', encoding='utf-8') as fin:
        with open(args.emb_output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                word = line.strip()
                if word in model:
                    embs = list(model[word])
                else:
                    embs = [1e-12 for _ in range(model.vector_size)]
                embs = [str(item) for item in embs]
                fout.write(' '.join([word]+embs)+'\n')

    if args.w2v_emb_output_path != '':
        glove2word2vec(args.emb_output_path, args.w2v_emb_output_path)


