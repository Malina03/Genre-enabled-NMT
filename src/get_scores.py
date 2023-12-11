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
bleu_df = pd.DataFrame(columns=['model', 'test_file', 'bleu'])
comet_df = pd.DataFrame(columns=['model', 'test_file', 'comet'])
# print(eval_files)

for f in eval_files:
    # print(f.split('/')[8])
    f_name = f.split('/')[8].split('_')[0]
    model = f.split('/')[7]
    # make a dictionary with scores by model and test file
    
    if 'bleu' in f and 'bleurt' not in f:
       
        bleu = float(open(f, "r").readlines()[0].strip('\n'))
        bleu_df = bleu_df.append({'model': model, 'test_file': f_name, 'bleu': bleu}, ignore_index=True)
    elif 'comet' in f:
        comet = [float(l.split(" ")[-1].strip()) for l in open(f, "r").readlines()]
        comet_df = comet_df.append({'model': model, 'test_file': f_name, 'comet': comet}, ignore_index=True)
# print(scores)
# make a dataframe with the scores

# merge the two dataframes by model and test file
# df = pd.merge(bleu_df, comet_df, on=['model', 'test_file'], how='outer')

# save the dataframe to a csv file
# df.to_csv('/scratch/s3412768/genre_NMT/en-hr/results/eval_scores.csv', index=False)
# save only bleu scores
bleu_df.to_csv('/scratch/s3412768/genre_NMT/en-hr/results/eval_bleu_scores.csv', index=False)
