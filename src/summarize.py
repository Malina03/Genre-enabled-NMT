from prettytable import PrettyTable
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Args to summarize evaluation scores.')
parser.add_argument('--folder', type=str, help='the folder where the evaluation files are stored')
parser.add_argument('--fname', type=str, help='the name of the file with predictions')
parser.add_argument('--ref_with_tags', type=str, help='the path to the reference file with tags (to determine the genres)')

def read_scores(folder, fname, ref_with_tags):
    ''' Reads the scores from the evaluation files and returns a dataframe with the scores and the genres. 
    
    Args:
    - folder (str): the folder where the evaluation files are stored
    - fname (str): the name of the file with predictions
    - ref_with_tags (str): the path to the reference file with tags (to determine the genres)
    
    Returns:
    - res (pandas DataFrame): a dataframe with the scores and the genres'''


    ref_with_tags = pd.read_csv(ref_with_tags, sep='\t', header=0)
    tokens_to_genres = {'>>info<<': 'Information/Explanation', '>>promo<<': 'Promotion', '>>news<<': 'News', '>>law<<': 'Legal', '>>other<<': 'Other', '>>arg<<': 'Opinion/Argumentation', '>>instr<<': 'Instruction', '>>lit<<': 'Prose/Lyrical', '>>forum<<': 'Forum'}
    genres = [tokens_to_genres[line.split(' ')[0]] for line in ref_with_tags.en_par_tokens.to_list()]
    
    bleu = float(open(folder + fname + "_predictions.txt.eval.bleu", "r").readlines()[0].strip('\n'))
    ter = float(open(folder + fname + "_predictions.txt.eval.ter", "r").readlines()[0].strip('\n'))
    chrf = float(open(folder + fname + "_predictions.txt.eval.chrf", "r").readlines()[0].strip('\n'))
    chrfpp = float(open(folder + fname + "_predictions.txt.eval.chrfpp", "r").readlines()[0].strip('\n'))
    bleurt = [float(l) for l in open(folder + fname + "_predictions.txt.eval.bleurt", "r").readlines()]
    # 2,4,6 precison, recall, f1
    bert_score = open(folder + fname + '_predictions.txt.eval.bertscore', 'r').readlines()[0].strip('\n').split(' ')
    comet = [float(l.split(" ")[-1].strip()) for l in open(folder + fname + "_predictions.txt.eval.comet", "r").readlines()]

    res = pd.DataFrame()
    res['genre'] = genres
    res['bert_score_f1'] = [float(bert_score[6])] * len(genres)
    res['bert_score_p'] = [float(bert_score[2])] * len(genres)
    res['bert_score_r'] = [float(bert_score[4])] * len(genres)
    res['bleu'] = [bleu] * len(genres)
    res['ter'] = [ter] * len(genres)
    res['chrf'] = [chrf] * len(genres)
    res['chrfpp'] = [chrfpp] * len(genres)
    res['comet_avg'] = [comet[-1]] * len(genres)
    res['comet'] = comet[1:-1]
    res['bleurt'] = bleurt[1:]

    return res


args = parser.parse_args()

res = read_scores(args.folder, args.fname, args.ref_with_tags)

x = PrettyTable()
x.field_names = ["Test file", "BLEU", "COMET", "TER", "chrF", "chrFpp", "BertScore_f1", "BertScore_precision", "BertScore_recall"]
row = [args.fname, round(res['bleu'].iloc[0],3), round(res['comet_avg'].iloc[0],3), round(res['ter'].iloc[0],3), round(res['chrf'].iloc[0],3), round(res['chrfpp'].iloc[0],3), round(res['bert_score_f1'].iloc[0],3), round(res['bert_score_p'].iloc[0],3), round(res['bert_score_r'].iloc[0],3)]
x.add_row(row)

y = PrettyTable()
y.add_column("Genre", list(res.groupby('genre').groups.keys()))
y.add_column('BLEURT', [round(s,3) for s in res.groupby('genre').mean()['bleurt'].to_list()])
y.add_column('COMET', [round(s,3) for s in res.groupby('genre').mean()['comet'].to_list()])
y.add_column('Count', [round(s,3) for s in res.groupby('genre').count()['bleurt'].to_list()])


with open(args.folder + args.fname + '_summary.txt', 'w') as w:
    w.write(str(x))
    w.write("\n\n")
    w.write(str(y.get_string(sortby="Count", reversesort=True)))

