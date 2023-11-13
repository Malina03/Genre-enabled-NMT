import pandas as pd


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
        genre_indices = dat[dat['en_doc'].str.startswith(f'>>{genres[0]}<<')].index.tolist()
        for genre in genres[1:]:
            genre_indices += dat[dat['en_doc'].str.startswith(f'>>{genre}<<')].index.tolist()
        # write to tsv only the lines with each genre
        dat.iloc[genre_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.{"_".join(genres)}.tag.tsv', sep='\t', index=False, header=False)
        del dat 
        # use the same indices to get the lines without the tags
        dat_no_tags = pd.read_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.tsv', sep='\t')
        dat_no_tags.iloc[genre_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.{"_".join(genres)}.tsv', sep='\t', index=False, header=False)
        del dat_no_tags

def main():
    make_multiple_genre_dataset('hr', ['train', 'dev'], ['law', 'news', 'lit'])
    
if __name__ == "__main__":
    main()