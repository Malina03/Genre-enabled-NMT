import pandas as pd

def main():
    language = 'hr'
    sets = ['train', 'dev']

    for s in sets:
        dat = pd.read_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.tag.tsv', sep='\t')
        # add column names to the dataframe
        dat.columns = ['en_doc', f'{language}_doc']
        # get the indices of the lines that start with >>info<<
        # info_indices = dat[dat['en_doc'].str.startswith('>>info<<')].index.tolist()
        # promo_indices = dat[dat['en_doc'].str.startswith('>>promo<<')].index.tolist()
        # instr_indices = dat[dat['en_doc'].str.startswith('>>instr<<')].index.tolist()
        # news_indices = dat[dat['en_doc'].str.startswith('>>news<<')].index.tolist()
        law_indices = dat[dat['en_doc'].str.startswith('>>law<<')].index.tolist()
        lit_indices = dat[dat['en_doc'].str.startswith('>>lit<<')].index.tolist()
        arg_indices = dat[dat['en_doc'].str.startswith('>>arg<<')].index.tolist()

        # write to tsv only the lines with each genre
        # dat.iloc[info_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.info.tag.tsv', sep='\t', index=False, header=False)
        # dat.iloc[promo_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.promo.tag.tsv', sep='\t', index=False, header=False)
        # dat.iloc[instr_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.instr.tag.tsv', sep='\t', index=False, header=False)
        # dat.iloc[news_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.news.tag.tsv', sep='\t', index=False, header=False)
        dat.iloc[law_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.law.tag.tsv', sep='\t', index=False, header=False)
        dat.iloc[lit_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.lit.tag.tsv', sep='\t', index=False, header=False)
        dat.iloc[arg_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.arg.tag.tsv', sep='\t', index=False, header=False)
        del dat 
        # use the same indices to get the lines without the tags

        dat_no_tags = pd.read_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.tsv', sep='\t')
        # dat_no_tags.iloc[info_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.info.tsv', sep='\t', index=False, header=False)
        # dat_no_tags.iloc[promo_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.promo.tsv', sep='\t', index=False, header=False)
        # dat_no_tags.iloc[instr_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.instr.tsv', sep='\t', index=False, header=False)
        # dat_no_tags.iloc[news_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.news.tsv', sep='\t', index=False, header=False)
        dat_no_tags.iloc[law_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.law.tsv', sep='\t', index=False, header=False)
        dat_no_tags.iloc[lit_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.lit.tsv', sep='\t', index=False, header=False)
        dat_no_tags.iloc[arg_indices].to_csv(f'/scratch/s3412768/genre_NMT/en-{language}/data/MaCoCu.en-{language}.{s}.arg.tsv', sep='\t', index=False, header=False)
        del dat_no_tags
    
if __name__ == "__main__":
    main()