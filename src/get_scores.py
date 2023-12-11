import os
import pandas as pd

root_dir = '/scratch/s3412768/genre_NMT/en-hr/eval/from_scratch/'

# find all evaluation files .eval.bleu, eval.comet
eval_files = []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.eval.bleu') or file.endswith('.eval.comet'):
            eval_files.append(os.path.join(root, file))

# get the scores from the files
scores = {}
# print(eval_files)

for f in eval_files:
    # print(f.split('/')[8])
    f_name = f.split('/')[8].split('_')[0]
    model = f.split('/')[7]
    # make a dictionary with scores by model and test file
    if model not in scores:
        scores[model] = {}
    scores[model][f_name] = {}
    if 'bleu' in f and 'bleurt' not in f:
        bleu = float(open(f, "r").readlines()[0].strip('\n'))
        scores[model][f_name]['bleu'] = bleu
    elif 'comet' in f:
        comet = [float(l.split(" ")[-1].strip()) for l in open(f, "r").readlines()]
        scores[model][f_name]['comet'] = comet

print(scores)
# make a dataframe with the scores
df = pd.DataFrame()

for model in scores:
    for f_name in scores[model]:
        # if value is missing, add NaN
        if 'bleu' not in scores[model][f_name]:
            scores[model][f_name]['bleu'] = float('NaN')
        if 'comet' not in scores[model][f_name]:
            scores[model][f_name]['comet'] = float('NaN')
        row = [model, f_name, scores[model][f_name]['bleu'], scores[model][f_name]['comet']]
        df.append(row)
df.columns = ['model', 'test_file', 'bleu', 'comet']
# save as csv file in /scratch/s3412768/genre_NMT/en-hr/results/
df.to_csv('/scratch/s3412768/genre_NMT/en-hr/results/eval_scores.csv', sep='\t', index=False, header=True)
