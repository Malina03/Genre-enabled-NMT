import pandas as pd
import os 

'''Several functions to make genre-specific datasets and balance their sizes, 
    from the MaCoCu corpus (processed using the pipeline in the preprocessing file).'''

def change_genre_tokens():
    data_folder = "/scratch/s3412768/genre_NMT/en-hr/data/"

    # get all files in data_folder/old_tokens

    files = os.listdir(data_folder + "old_tokens/")

    for f in files:
        # read in the file
        df = pd.read_csv(data_folder + "old_tokens/" + f, sep="\t", header=None, names=["source", "target"])
        # replace >> with < and << with >
        df["source"] = df["source"].str.replace(">>", "<")
        df["source"] = df["source"].str.replace("<<", ">")
        # write the file
        df.to_csv(data_folder + f, sep="\t", header=None, index=False)



def make_single_genre_dataset(language, sets, genre):
    for s in sets:
        dat = pd.read_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.tag.tsv', sep='\t')
        # add column names to the dataframe
        dat.columns = ['en_doc', f'{language}_doc']
        # get the indices of the lines that start with the genre tag
        # genre_indices = dat[dat['en_doc'].str.startswith(f'>>{genre}<<')].index.tolist()
        genre_indices = dat[dat['en_doc'].str.startswith(f'<{genre}>')].index.tolist()
        # write to tsv only the lines with each genre
        dat.iloc[genre_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.{genre}.tag.tsv', sep='\t', index=False, header=False)
        del dat 
        # use the same indices to get the lines without the tags
        dat_no_tags = pd.read_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.tsv', sep='\t')
        dat_no_tags.iloc[genre_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.{genre}.tsv', sep='\t', index=False, header=False)
        del dat_no_tags

def make_multiple_genre_dataset(language, sets, genres):
    for s in sets:
        dat = pd.read_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.tag.tsv', sep='\t')
        # add column names to the dataframe
        dat.columns = ['en_doc', f'{language}_doc']
        # get the indices of the lines that start with the genre tag
        # genre_indices = dat[dat['en_doc'].str.startswith(f'>>{genres[0]}<<')].index.tolist()
        genre_indices = dat[dat['en_doc'].str.startswith(f'<{genres[0]}>')].index.tolist()
        for genre in genres[1:]:
            # genre_indices += dat[dat['en_doc'].str.startswith(f'>>{genre}<<')].index.tolist()
            genre_indices += dat[dat['en_doc'].str.startswith(f'<{genre}>')].index.tolist()
        # write to tsv only the lines with each genre
        dat.iloc[genre_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.{"_".join(genres)}.tag.tsv', sep='\t', index=False, header=False)
        del dat 
        # use the same indices to get the lines without the tags
        dat_no_tags = pd.read_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.tsv', sep='\t')
        dat_no_tags.iloc[genre_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.{"_".join(genres)}.tsv', sep='\t', index=False, header=False)
        del dat_no_tags


def make_balanced_datasets(language, genres):
    sets = ['train', 'dev']
    genre_tokens = {'Prose/Lyrical': '<lit>','Instruction': '<instr>', 'Promotion': '<promo>', 'Opinion/Argumentation': '<arg>' , 'Other': '<other>' , 'Information/Explanation': '<info>', 'News': '<news>', 'Legal': '<law>', 'Forum': '<forum>'}
    genre_abv = {g: genre_tokens[g][1:-1] for g in genres}
    all_data = pd.read_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}_complete.tsv', sep='\t', header=0)
    # only keep en_doc, hr_doc, X-genre and set columns
    all_data = all_data[['en_par', f'{language}_par', 'set', 'X-GENRE']]
    # remove test set
    print(all_data.set.unique())
    all_data = all_data[all_data['set'] != 'test']
    only_req_genres = all_data[all_data['X-GENRE'].isin(genres)]
    remaining_genres = all_data[~all_data['X-GENRE'].isin(genres)]
    # get the minimum number of lines per genre per set to dictionary
    print(only_req_genres.groupby(['set', 'X-GENRE'])['en_par'].count())
    min_examples = only_req_genres.groupby(['set', 'X-GENRE'])['en_par'].count().groupby('set').min().to_dict()
    print(min_examples)
    for s in sets:
        for g in genres:
            # randomly sample the required number of lines per set and genre
            sampled = only_req_genres[only_req_genres['set'] == s][only_req_genres['X-GENRE'] == g].sample(n=min_examples[s])
            # save en_par and hr_par to tsv
            sampled[['en_par', f'{language}_par']].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.{genre_abv[g]}.tsv', sep='\t', index=False, header=False, quoting=3)
            # write to tsv adding the genre tag in the beginning of each en_par
            sampled['en_par'] = genre_tokens[g] + ' ' + sampled['en_par'].astype(str)
            sampled[['en_par', f'{language}_par']].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.{genre_abv[g]}.tag.tsv', sep='\t', index=False, header=False, quoting=3)
            del sampled
    # randomly sample the required number of lines for a random genre dataset balanced wrt the number of lines per set
    for s in sets:
        sampled = all_data[all_data['set'] == s].sample(n=min_examples[s])
        # print to a file the number of lines per genre
        sampled.groupby('X-GENRE')['en_par'].count().to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.random.counts.log', sep='\t', index=True, header=True)
        # save en_par and hr_par to tsv
        sampled[['en_par', f'{language}_par']].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.random.tsv', sep='\t', index=False, header=False, quoting=3)
        # write to tsv adding the genre tag in the beginning of each en_par
        # add the genre tag in the beginning of each en_par
        sampled['en_par'] = sampled['X-GENRE'].apply(lambda x: genre_tokens[x]) + ' ' + sampled['en_par'].astype(str)
        sampled[['en_par', f'{language}_par']].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.random.tag.tsv', sep='\t', index=False, header=False, quoting=3)
        del sampled
                                                                                                       


def main():
    make_balanced_datasets('tr', ['Legal', 'News', 'Promotion', 'Information/Explanation'])
    
if __name__ == "__main__":
    main()