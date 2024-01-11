import os
import pandas as pd

root_dir = '/scratch/s3412768/genre_NMT/en-hr/eval/from_scratch/'
root_data_dir = '/scratch/s3412768/genre_NMT/en-hr/data/'

# find all evaluation files .eval.bleu, eval.comet
eval_files = []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.eval.bleu') or file.endswith('.eval.comet'):
            eval_files.append(os.path.join(root, file))

# get the scores from the files
bleu_df = pd.DataFrame(columns=['model', 'test_file', 'bleu'])
comet_df = pd.DataFrame(columns=['model', 'test_file', 'comet'])
comet_scores_per_genre = pd.DataFrame(columns=['model', 'test_file', 'genre', 'comet_mean', 'comet_std'])
list_of_df = []
# print(eval_files)

tokens_to_genres = {'<info>': 'Information/Explanation', '<promo>': 'Promotion', '<news>': 'News', '<law>': 'Legal', '<other>': 'Other', '<arg>': 'Opinion/Argumentation', '<instr>': 'Instruction', '<lit>': 'Prose/Lyrical', '<forum>': 'Forum'}

ref_with_tags_macocu = pd.read_csv(root_data_dir + 'MaCoCu.en-hr.test.tag.tsv', sep='\t', header=None)
genres_macocu = [tokens_to_genres[line.split(' ')[0]] for line in ref_with_tags_macocu[ref_with_tags_macocu.columns[0]].to_list()]
ref_with_tags_floresdev = pd.read_csv(root_data_dir + 'floresdev.en-hr.test.tag.tsv', sep='\t', header=None)
genres_floresdev = [tokens_to_genres[line.split(' ')[0]] for line in ref_with_tags_floresdev[ref_with_tags_floresdev.columns[0]].to_list()]
ref_with_tags_floresdevtest = pd.read_csv(root_data_dir + 'floresdevtest.en-hr.test.tag.tsv', sep='\t', header=None)
genres_floresdevtest = [tokens_to_genres[line.split(' ')[0]] for line in ref_with_tags_floresdevtest[ref_with_tags_floresdevtest.columns[0]].to_list()]
# ref_with_tags_wmttest2022 = pd.read_csv(root_data_dir + 'wmttest2022.en-hr.test.tag.tsv', sep='\t', header=None)
# genres_wmttest2022 = [tokens_to_genres[line.split(' ')[0]] for line in ref_with_tags_wmttest2022[ref_with_tags_wmttest2022.columns[0]].to_list()]


for f in eval_files:
    # print(f.split('/')[8])
    f_name = f.split('/')[8].split('_')[0]
    model = f.split('/')[7]
    # make a dictionary with scores by model and test file
    
    if 'bleu' in f and 'bleurt' not in f:
       
        bleu = float(open(f, "r").readlines()[0].strip('\n'))
        bleu_df = bleu_df.append({'model': model, 'test_file': f_name, 'bleu': bleu}, ignore_index=True)
    elif 'comet' in f:
        # get the score in the last line
        try :
            comet = float(open(f, "r").readlines()[-2].split(" ")[-1].strip())
        except:
            print(f)
            continue
        comet = float(open(f, "r").readlines()[-1].split(" ")[-1].strip())
        comet_df = comet_df.append({'model': model, 'test_file': f_name, 'comet': comet}, ignore_index=True)
        # get scores per sentence
        individial_scores_comet = [float(l.split(" ")[-1].strip()) for l in open(f, "r").readlines()[:-1]]
        if f_name == "MaCoCu":
            genres = genres_macocu
        elif f_name == "floresdev":
            genres = genres_floresdev
        elif f_name == "floresdevtest":
            genres = genres_floresdevtest
        # elif "wmttest2022" in f_name:
        #     genres = genres_wmttest2022
        else:
            print("Error: test file not recognized")
        # print(len(genres))
        # print(len(individial_scores_comet))
        scores_per_genre = pd.DataFrame({'model': [model]*len(genres), 'test_file': [f_name]*len(genres), 'genre': genres, 'comet': individial_scores_comet})
        # comet_scores_per_genre = comet_scores_per_genre.append(scores_per_genre, ignore_index=True)
        # compute the average score per genre and standard deviation
        scores_per_genre = scores_per_genre.groupby(['model', 'test_file', 'genre']).agg({'comet': ['mean', 'std']}).reset_index()
        scores_per_genre.columns = ['model', 'test_file', 'genre', 'comet_mean', 'comet_std']
        list_of_df.append(scores_per_genre)


# print(scores)
# make a dataframe with the scores

# merge the two dataframes by model and test file
# df = pd.merge(bleu_df, comet_df, on=['model', 'test_file'], how='outer')

# save the dataframe to a csv file
# df.to_csv('/scratch/s3412768/genre_NMT/en-hr/results/eval_scores.csv', index=False)
# save only bleu scores
bleu_df.to_csv('/scratch/s3412768/genre_NMT/en-hr/results/eval_bleu_scores.csv', index=False)
# save only comet scores
comet_df.to_csv('/scratch/s3412768/genre_NMT/en-hr/results/eval_comet_scores.csv', index=False)
# save comet scores per genre
comet_scores_per_genre = pd.concat(list_of_df)
comet_scores_per_genre.to_csv('/scratch/s3412768/genre_NMT/en-hr/results/eval_comet_scores_per_genre.csv', index=False)
