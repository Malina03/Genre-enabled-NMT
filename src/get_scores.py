import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_folder', type=str, default='from_scratch', help='path to the folder with the experiments')
parser.add_argument('--language', type=str, default='hr', help='target language')

args = parser.parse_args()


root_dir = '/scratch/s3412768/genre_NMT/en-' + args.language + '/eval/' + args.exp_folder + '/'
root_data_dir = '/scratch/s3412768/genre_NMT/en-' + args.language + '/data/'

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

ref_with_tags_macocu = pd.read_csv(root_data_dir + 'MaCoCu.en-'+ args.language + '.test.tag.tsv', sep='\t', header=None)
genres_macocu = [tokens_to_genres[line.split(' ')[0]] for line in ref_with_tags_macocu[ref_with_tags_macocu.columns[0]].to_list()]
ref_with_tags_floresdev = pd.read_csv(root_data_dir + 'floresdev.en-'+ args.language + '.test.tag.tsv', sep='\t', header=None)
genres_floresdev = [tokens_to_genres[line.split(' ')[0]] for line in ref_with_tags_floresdev[ref_with_tags_floresdev.columns[0]].to_list()]
ref_with_tags_floresdevtest = pd.read_csv(root_data_dir + 'floresdevtest.en-'+ args.language + '.test.tag.tsv', sep='\t', header=None)
genres_floresdevtest = [tokens_to_genres[line.split(' ')[0]] for line in ref_with_tags_floresdevtest[ref_with_tags_floresdevtest.columns[0]].to_list()]
ref_with_targs_macocu_doc = pd.read_csv(root_data_dir + 'MaCoCu.en-'+ args.language + '.doc.test.tag.tsv', sep='\t', header=None)
genres_macocu_doc = [tokens_to_genres[line.split(' ')[0]] for line in ref_with_targs_macocu_doc[ref_with_targs_macocu_doc.columns[0]].to_list()]

if args.language == 'hr':
    ref_with_tags_wmttest2022 = pd.read_csv(root_data_dir + 'wmttest2022.en-hr.test.tag.tsv', sep='\t', header=None)
    genres_wmttest2022 = [tokens_to_genres[line.split(' ')[0]] for line in ref_with_tags_wmttest2022[ref_with_tags_wmttest2022.columns[0]].to_list()] 
elif args.language == 'is':
    ref_with_tags_wmttest2022 = pd.read_csv(root_data_dir + 'wmttest2021.en-is.test.tag.tsv', sep='\t', header=None)
    genres_wmttest2022 = [tokens_to_genres[line.split(' ')[0]] for line in ref_with_tags_wmttest2022[ref_with_tags_wmttest2022.columns[0]].to_list()]
elif args.language == 'tr':
    ref_with_tags_wmttest2022 = pd.read_csv(root_data_dir + 'wmttest2018.en-tr.test.tag.tsv', sep='\t', header=None)
    genres_wmttest2022 = [tokens_to_genres[line.split(' ')[0]] for line in ref_with_tags_wmttest2022[ref_with_tags_wmttest2022.columns[0]].to_list()]
else:
    print("Language not recognized")
    
for f in eval_files:
    # print(f.split('/')[8])
    f_name = f.split('/')[8].split('_')[0]
    model = f.split('/')[7]
    # make a dictionary with scores by model and test file
    # if model name doesn't end in _1 or _2 or _3 skip - only include exp with seeds
    if model[-2] != '_' and args.exp_folder != 'opus':
        continue

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
        if "doc" in f:
            genres = genres_macocu_doc
        elif f_name == "MaCoCu":
            genres = genres_macocu
        elif f_name == "floresdev":
            genres = genres_floresdev
        elif f_name == "floresdevtest":
            genres = genres_floresdevtest
        elif "wmttest2022" in f_name:
            genres = genres_wmttest2022
        else:
            print("Error: test file not recognized")
        print("test file: {} and model: {}".format(f_name, model))
        print(len(genres))
        print(len(individial_scores_comet))
        scores_per_genre = pd.DataFrame({'model': [model]*len(genres), 'test_file': [f_name]*len(genres), 'genre': genres, 'comet': individial_scores_comet})
        comet_scores_per_genre = comet_scores_per_genre.append(scores_per_genre, ignore_index=True)
        # compute the average score per genre and standard deviation
        scores_per_genre = scores_per_genre.groupby(['model', 'test_file', 'genre']).agg({'comet': ['mean', 'std']}).reset_index()
        scores_per_genre.columns = ['model', 'test_file', 'genre', 'comet_mean', 'comet_std']
        # split model name in model and seed
        scores_per_genre['seed'] = scores_per_genre['model'].str[-1]
        scores_per_genre['model'] = scores_per_genre['model'].str[:-2]
        # compute the average score per model and standard deviation
        scores_per_model = scores_per_genre.groupby(['model', 'test_file', 'genre']).agg({'comet_mean': ['mean', 'std']}).reset_index()
        scores_per_model.columns = ['model', 'test_file', 'genre', 'comet_mean', 'comet_std']
        list_of_df.append(scores_per_genre)


# print(scores)
# make a dataframe with the scores

# merge the two dataframes by model and test file
df = pd.merge(bleu_df, comet_df, on=['model', 'test_file'], how='outer')

# save the dataframe to a csv file
df.to_csv(f'/scratch/s3412768/genre_NMT/en-hr/results/{args.exp_folder}_eval_scores.csv', index=False)
# split bleu model name in model and seed
bleu_df['seed'] = bleu_df['model'].str[-1]
bleu_df['model'] = bleu_df['model'].str[:-2]
# split comet model name in model and seed
comet_df['seed'] = comet_df['model'].str[-1]
comet_df['model'] = comet_df['model'].str[:-2]
# compute the average score per model and standard deviation
bleu_df = bleu_df.groupby(['model', 'test_file']).agg({'bleu': ['mean', 'std']}).reset_index()
bleu_df.columns = ['model', 'test_file', 'bleu_mean', 'bleu_std']
comet_df = comet_df.groupby(['model', 'test_file']).agg({'comet': ['mean', 'std']}).reset_index()
comet_df.columns = ['model', 'test_file', 'comet_mean', 'comet_std']
bleu_df.to_csv('/scratch/s3412768/genre_NMT/en-' + args.language + '/results/' + args.exp_folder + '_bleu_scores.csv', index=False)
# save only comet scores
comet_df.to_csv('/scratch/s3412768/genre_NMT/en-' + args.language + '/results/' + args.exp_folder + '_comet_scores.csv', index=False)
# save comet scores per genre
comet_scores_per_genre = pd.concat(list_of_df)
comet_scores_per_genre.to_csv('/scratch/s3412768/genre_NMT/en-' + args.language + '/results/' + args.exp_folder + '_comet_scores_per_genre.csv', index=False)
