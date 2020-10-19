from gensim.models import KeyedVectors
import numpy as np
import sys
import tqdm
import copy
import argparse

if __name__ == '__main__':
    '''
    '''

    parser = argparse.ArgumentParser(description='Embedding similarity order generation')
    parser.add_argument("--w2v_emb_path", type=str, default="")
    parser.add_argument("--d2gpo_order_txt", type=str, default="")
    parser.add_argument("--d2gpo_order_idx", type=str, default="")
    args = parser.parse_args()

    model = KeyedVectors.load_word2vec_format(args.w2v_emb_path)

    if args.d2gpo_order_txt != "":
        fout1 = open(args.d2gpo_order_txt, 'w', encoding='utf-8')
    
    with open(args.d2gpo_order_idx, 'w', encoding='utf-8') as fout2:
        for idx in tqdm.tqdm(range(len(model.vocab))):
            word = model.index2word[idx]
            most_similar = model.most_similar(word, topn=None)
            most_similar_index = np.argsort(-most_similar)
            most_similar_words = [model.index2word[widx] for widx in list(most_similar_index)]
            most_similar_index = list(most_similar_index)
            if args.d2gpo_order_txt != "":
                fout1.write(' '.join(most_similar_words)+'\n')
            fout2.write(' '.join([str(item) for item in most_similar_index])+'\n')
    
    if args.d2gpo_order_txt != "":
        fout1.close()


