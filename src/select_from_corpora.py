import pandas as pd
from x_genre_predict import classify_dataset

def select_lines(file_name, out_file_name):
    ''' Selects lines from the corpus that have more than 25 words and saves them to a new file. '''
    corpus_src = []
    corpus_tgt = []
    error_count = 0
    with open(file_name, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                src, tgt = line.strip().split('\t')
                # only select lines with more than 25 words
                if len(src.split()) < 25:
                    continue
                corpus_src.append(src)
                corpus_tgt.append(tgt)
            except:
                error_count += 1
                continue

    print("Number of lines with errors:", error_count)
    print("Number of lines in the corpus:", len(corpus_src))

    # save the corpus to a file in the same format as the original corpus
    with open(out_file_name, "w") as f:
        for i in range(len(corpus_src)):
            f.write("{}\t{}\n".format(corpus_src[i], corpus_tgt[i]))

def get_genre_labels(file_name, tgt_language, save_file):
    data = pd.read_csv(file_name, sep='\t', header=None)
    # add column names to the dataframe
    data.columns = ['en_doc', f'{tgt_language}_doc']
    df_labelled = classify_dataset(data, 'en_doc', save_file)
    return df_labelled

if __name__ == "__main__":
    df_labelled = get_genre_labels('/scratch/hb-macocu/NMT_eval/en-is/data/CCM.CCA.Para.Tilde.en-is.dedup.norm.tsv', 'is', '/scratch/hb-macocu/NMT_eval/en-is/data/CCM.CCA.Para.Tilde.en-is.25.labels.dedup.norm.tsv')
    