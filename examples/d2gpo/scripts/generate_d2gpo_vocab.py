import argparse

if __name__ == '__main__':
    '''
    '''

    parser = argparse.ArgumentParser(description='D2GPo vocab preparation')
    parser.add_argument("--fairseq_vocab_path", type=str, default="")
    parser.add_argument("--d2gpo_vocab_path", type=str, default="")
    args = parser.parse_args()

    # NOTES: This only applies to the NMT implementation in Fairseq. 
    # When you use a new implementation or new task, please change it to what you need.
    specials = ['<s>', '<pad>', '</s>', '<unk>']

    with open(args.fairseq_vocab_path, 'r', encoding='utf-8') as fin:
        with open(args.d2gpo_vocab_path, 'w', encoding='utf-8') as fout:
            for word in specials:
                fout.write(word+'\n')
            for line in fin:
                idx = line.rfind(' ')
                word = line[:idx]
                fout.write(word+'\n')

