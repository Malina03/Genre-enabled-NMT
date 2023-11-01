import gzip
import shutil
# import wget
import regex as re
import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
from x_genre_predict import classify_dataset
import os


def download_corpus(url, f_name):
    # Downloading the file by sending the request to the URL
    corpus_file = wget.download(url, f_name)
    print('Downloading Completed')

    
def unzip_corpus(f_name): 
# Unzip the file
    with gzip.open(f_name, 'rb') as f_in:
        with open(f_name[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print('Unzipping Completed')
    return 

def tmx_to_json(fname, tgt_language, save_path):

    if not Path(fname).exists():
        print(f"Unzipping file {fname}.gz")
        unzip_corpus(Path(fname.name + ".gz"))
        # Read the corpus

    corpus = open(fname, "rb").read()
    corpus = corpus.decode("utf-8")

    if tgt_language == "hr":
        lang_id = "hr_latin"

    tu_re = re.compile('<tu tuid=".*?>\n(.*?)<\/tu>', re.DOTALL)
    # Compile relevant information inside tus
    bi_score_re = re.compile('<prop type="score-bicleaner-ai">(.*?)</prop>')
    # biroamer_re = re.compile('<prop type="biroamer-entities">(.*?)</prop>')
    translation_dir_re = re.compile('<prop type="translation-direction">(.*?)</prop>')
    en_source_re = re.compile('<tuv xml:lang="en">.*?<prop type="source-document">(.*?)</prop>', re.DOTALL)
    en_par_id_re = re.compile('<tuv xml:lang="en">.*?<prop type="paragraph-id">(.*?)</prop', re.DOTALL)
    en_par_re = re.compile('<tuv xml:lang="en">.*?<seg>(.*?)</seg>', re.DOTALL)
    en_var_doc_re = re.compile('<prop type="english-variant-document">(.*?)</prop>')
    en_var_dom_re = re.compile('<prop type="english-variant-domain">(.*?)</prop>')
    sl_source_re = re.compile(f'<tuv xml:lang="{lang_id}">.*?<prop type="source-document">(.*?)</prop>', re.DOTALL)
    sl_par_id_re = re.compile(f'<tuv xml:lang="{lang_id}">.*?<prop type="paragraph-id">(.*?)</prop', re.DOTALL)
    sl_par_re = re.compile(f'<tuv xml:lang="{lang_id}">.*?<seg>(.*?)</seg>', re.DOTALL)
    tus_list = tu_re.findall(corpus)
    
    tus_content = []
    for i in tus_list:
        # Find all relevant information based on regexes
        bi_score = bi_score_re.search(i).group(1)
        # biroamer = biroamer_re.search(i).group(1)
        translation_dir = translation_dir_re.search(i).group(1)
        en_source = en_source_re.search(i).group(1)
        en_par_id = en_par_id_re.search(i).group(1)
        en_par = en_par_re.search(i).group(1)
        en_var_doc = en_var_doc_re.search(i).group(1)
        en_var_dom = en_var_dom_re.search(i).group(1)
        sl_source = sl_source_re.search(i).group(1)
        sl_par_id = sl_par_id_re.search(i).group(1)
        sl_par = sl_par_re.search(i).group(1)
        # Add information to the dictionary
        current_tu = {"score_bicleaner_ai": float(bi_score), "translation_direction": translation_dir, "en_source": en_source, "en_par_id": en_par_id, "en_par": en_par, "en_var_doc": en_var_doc, "en_var_dom": en_var_dom, f"{tgt_language}_source": sl_source, f"{tgt_language}_par_id": sl_par_id, f"{tgt_language}_par": sl_par}
        # Append the dictionary to the list
        tus_content.append(current_tu)
        
    with open(save_path, "w") as file:
        json.dump(tus_content,file, indent= "")

def preprocess(path, lang_code, length_threshold, drop_par_duplicates = True, drop_doc_duplicates = True, keep_columns=False, info = True):
	"""
	Takes the JSON file name, created in the tmx_to_json function,
	transforms it into a pandas DataFrame, preprocesses it
	and saves the final document-level CSV file to which filter_non_textual function is to be applied.

	Args:
	- file name (str): the path to the JSON file
	- lang code: the code of the language that is in the pair with English,	it is the same as in the name of the MaCoCu file (e.g., mk in MaCoCu-mk-en)
	"""


	with open(path/f"MaCoCu-{lang_code}-en.json", "r") as file:
		tus_content = json.load(file)

	# Convert data to a dataframe
	corpus_df = pd.DataFrame(tus_content)
    
	print("Dataframe loaded. Initial number of sentences: {}".format(corpus_df.en_source.count()))
	# Sort by english url and then by en_par_id to order the paragraphs into texts
	corpus_df = corpus_df.sort_values(by = ["en_source", "en_par_id"])

	# Add information about domains
	domain_re=re.compile(r'^https?://(?:www\.)?(.+?)[/$]')

	en_domain_list = [domain_re.search(i).group(1) for i in corpus_df.en_source.to_list()]

	corpus_df["en_domain"] = en_domain_list

	# Repeat with domain of the other language
	sl_domain_list = [domain_re.search(i).group(1) for i in corpus_df[f"{lang_code}_source"].to_list()]
	corpus_df[f"{lang_code}_domain"] = sl_domain_list

	# Add information whether the domains are the same
	corpus_df["same_domains"] = np.where(corpus_df["en_domain"] == corpus_df[f"{lang_code}_domain"], "yes", 'no')
	print("Added same domains column.")


	# # Add column for domains that are different
	# corpus_df["different_domains"] = corpus_df["en_domain"] + " " + corpus_df[f"{lang_code}_domain"]

	if info == True:
	# Print the information
		print("Information about the web domains for the two languages is added. See the head of the dataframe:\n")
		print(corpus_df.head(2))

		print("Number of same and different domains in the corpus:\n")

		print(corpus_df["same_domains"].value_counts().to_markdown())

		# Number of texts and sentences up to now
		previous_no_sentences = corpus_df.en_source.count()
		previous_no_texts = len(corpus_df.en_source.unique())
		print(f"\nCurrent number of sentences: {previous_no_sentences}")
		print(f"Current number of texts: {previous_no_texts}\n\n")

	# See number of discarded texts and sentences
	def calculate_discarded(previous_no_sentences, previous_no_texts, calculate_texts_only):
		new_number_sentences = corpus_df.en_source.count()
		new_number_texts = len(corpus_df.en_source.unique())
		if calculate_texts_only == False:
			print(f"New number of sentences: {new_number_sentences}")
			print(f"No. of discarded sentences: {previous_no_sentences-new_number_sentences}, percentage: {(previous_no_sentences-new_number_sentences)/previous_no_sentences}")
		
		print(f"New number of texts: {new_number_texts}")
		print(f"No. of discarded texts: {previous_no_texts-new_number_texts}, percentage: {(previous_no_texts-new_number_texts)/previous_no_texts}")

		return new_number_sentences, new_number_texts
	
	# Discard instances that are from different domains
	corpus_df = corpus_df[corpus_df["same_domains"] == "yes"]

	if info == True:
		print("Instances from different domains were discarded.\n")

		sentences_same_domains, texts_same_domains = calculate_discarded(previous_no_sentences, previous_no_texts, False)

	# Calculate average bicleaner ai score based on the en_source
	corpus_df["average_score"] = corpus_df["score_bicleaner_ai"].groupby(corpus_df['en_source']).transform('mean')

	if drop_par_duplicates == True:
	# Join par id and text
		corpus_df["en-par-src-text"] = corpus_df["en_par_id"] + "-" + corpus_df["en_source"] + "-" + corpus_df["en_par"]
		# Discard all duplicated English paragraphs with the same par id
		corpus_df = corpus_df.drop_duplicates("en-par-src-text")
		if info:
			print("\nAll duplicated English sentences with the same paragraph and sentence ID were discarded.\n")
			sentences_dupl_sent, text_dupl_sent = calculate_discarded(sentences_same_domains, texts_same_domains, False)
	else:
		sentences_dupl_sent = sentences_same_domains
		text_dupl_sent= texts_same_domains 
		
	# select unique sources from en and tgt language
	lang_to_en = corpus_df['en_source'].groupby(corpus_df[f'{lang_code}_source']).unique().to_dict()
	en_to_lang = corpus_df[f'{lang_code}_source'].groupby(corpus_df['en_source']).unique().to_dict()

	# select keys where the length of the list is 1 from the dictionary
	coresp_lang_en = {k: v[0] for k, v in en_to_lang.items() if len(v) == 1}
	coresp_en_lang = {k: v[0] for k, v in lang_to_en.items() if len(v) == 1}

	# find the en sources that correspond to a single is source and the other way around just to check
	common_lang = list(set(coresp_en_lang.keys()).intersection(list(coresp_lang_en.values())))
	common_en = list(set(coresp_lang_en.keys()).intersection(list(coresp_en_lang.values())))

	assert len(common_en) == len(common_lang)

	# Only keep sentences from docs that have the same unique source in both languages
	corpus_df = corpus_df[corpus_df['en_source'].isin(common_en) & corpus_df[f'{lang_code}_source'].isin(common_lang)]


	# Add to each instance from the same en_source joint text from all sentences
	corpus_df["en_doc"] = corpus_df["en_par"].groupby(corpus_df['en_source']).transform(' '.join)

	# Repeat with the text in other language
	corpus_df[f"{lang_code}_doc"] = corpus_df[f"{lang_code}_par"].groupby(corpus_df[f'{lang_code}_source']).transform(' '.join)


	if drop_doc_duplicates == True:
		# Keep only one example of each text
		corpus_df = corpus_df.drop_duplicates("en_doc")
		if info:
			print("\nThe sentences were merged into texts based on the source URL and the English duplicated texts were removed.\n")
			sentences_after_text_deduplication, texts_after_text_deduplication = calculate_discarded(sentences_dupl_sent, text_dupl_sent, True)

	# Add information about length
	corpus_df["en_length"] = corpus_df.en_doc.str.split().str.len()

	# Add information about length of the other language
	corpus_df[f"{lang_code}_length"] = corpus_df[f"{lang_code}_doc"].str.split().str.len()

	if info == True:
		print("\nInitial length of texts in the corpus:")
		print(corpus_df.en_length.describe().to_markdown())

	# Discard instances that have length less than  79 (median from other datasets)
	corpus_df = corpus_df[corpus_df["en_length"] > length_threshold - 1]

	if info == True:
		print(f"\nTexts that have less than {length_threshold} words were discarded.\n")

		sentences_after_length, texts_after_length = calculate_discarded(sentences_after_text_deduplication, texts_after_text_deduplication, True)

	corpus_df['length_diff'] = abs(corpus_df[f'{lang_code}_length'] - corpus_df['en_length'])

	if info == True:
		# Difference in length between documents
		print("\nDifference in length between documents:\n")
		print(corpus_df['length_diff'].describe().to_markdown())

	
	if keep_columns == False:
		corpus_df = corpus_df.drop(columns = ['score_bicleaner_ai', 'en_par_id', 'en_par', f'{lang_code}_par_id', f'{lang_code}_par',  'same_domains'])
		if drop_par_duplicates == True:
			corpus_df = corpus_df.drop(columns = ['en-par-src-text'])

	if info == True:
		# View the final dataframe
		print("The final dataframe: \n")
		print(corpus_df.head(5))
		
	if not drop_doc_duplicates:
		corpus_df.to_csv(path/f"Macocu-{lang_code}-en-doc-format-duplicates.csv", sep= "\t")
	else:
		corpus_df.to_csv(path/f"Macocu-{lang_code}-en-doc-format.csv", sep= "\t")	
	del corpus_df
    
	return 


def satisfy_all_genre_counts(target_cnt, curr_cnt, genres, data):
    """ Function to check if all genre counts are satisfied.
    Args:
        target_cnt (dict): target counts for each genre
        curr_cnt (dict): current counts for each genre
        genres (list): list of genres in a given domain
        data (pandas.DataFrame): data for a given domain

    Returns:
        bool: True if all genre counts are satisfied, False otherwise
    """

    for genre in genres:
        if curr_cnt[genre] + data[data['X-GENRE']== genre]['en_par'].sum() > target_cnt[genre]:
            return False
    return True



def satisfied_min_genre_count(target_cnt, curr_cnt, genres):
    for genre in genres:
        if curr_cnt[genre] < target_cnt[genre]:
            return False
    return True

def valid_addition(target_cnt, curr_cnt, genres, data):
    """ Function to check if a domain can be added to the test set.

    Args:


    """
    if satisfied_min_genre_count(target_cnt, curr_cnt, genres):
        return False
    
    for genre in genres:
        if curr_cnt[genre] + data[data['X-GENRE'] == genre]['en_par'].sum() > 3 * target_cnt[genre]:
            # check if for the rest of the genres the current_count + data would be greater than the target count 3 times
            return False
    return True

def update_genre_counts(curr_cnt, genres, data):
    """ Function to update genre counts.
    Args:
        curr_cnt (dict): current counts for each genre
        genres (list): list of genres in a given domain
        data (pandas.DataFrame): data for a given domain
    
    Returns:
        curr_cnt (dict): updated counts for each genre
    """
    for genre in genres:
        curr_cnt[genre] += data[data['X-GENRE'] == genre]['en_par'].sum()
    return curr_cnt

def check_non_zero(test_cnt, dev_cnt):
    """ Function to check if the target counts are non-zero.
    Args:
        test_target_cnt (dict): target counts for test set
        dev_target_cnt (dict): target counts for dev set
    """
    for genre in test_cnt:
        if test_cnt[genre] == 0:
            raise ValueError("Test count for genre {} is 0".format(genre))
    for genre in dev_cnt:
        if dev_cnt[genre] == 0:
            raise ValueError("Dev  count for genre {} is 0".format(genre))

def split_data(data, test_prop= 0.1, dev_prop = 0.1, test_size = 0, dev_size = 0, balance = True):
    """
    Split data into train, dev, and test sets. All splits have the same distribution of genres. 
    Internet domains of the data don't overlap bw splits.

    Args:
        data (pandas.DataFrame): data to split
        test_prop (float): proportion of data to put in test set. Default is 0.1
        dev_prop (float): proportion of data to put in dev set. Default is 0.1
        test_size (int): number of sentences to add to the test set. Default is 0. If it is greater than 0, 
            then the test_no is used instead of test_prop to determine the size of the test set.
        dev_size (int): number of sentences to add to the dev set. Default is 0. If it is greater than 0, 
            then the dev_no is used instead of dev_prop to determine the size of the dev set.
    
    Returns:
        train (pandas.DataFrame): train set
        dev (pandas.DataFrame): dev set
        test (pandas.DataFrame): test set
    """
    # print the number of sentences in each genre
    print(data.groupby(['X-GENRE'])['en_par'].count().reset_index())
    # remove sentences with genre 0
    data = data[data['X-GENRE'] != "0"]

    dom_genre = data.groupby(['en_domain','X-GENRE'])['en_par'].count().reset_index()
    labels = list(dom_genre['X-GENRE'].unique())
    print(labels)
    if balance:
        ratios = {label : dom_genre[dom_genre['X-GENRE']==label]['en_par'].sum()/dom_genre['en_par'].sum() for label in labels}
        total = dom_genre['en_par'].sum()
        if test_size != 0:
            test_prop = test_size/total
        if dev_size != 0:
            dev_prop = dev_size/total
        test_target_cnt = {label: int(ratios[label] * total * test_prop) for label in ratios}
        dev_target_cnt = {label: int(ratios[label] * total * dev_prop) for label in ratios}
    else:
         # have a fixed number of 1000 sentences per genre in each set
        test_target_cnt = {label: 1000 for label in labels}
        dev_target_cnt = {label: 1000 for label in labels}
    test_curr_cnt = {label: 0 for label in labels}
    dev_curr_cnt = {label: 0 for label in labels}
    test_domains = []
    dev_domains = []
    train_domains = []

    for domain in dom_genre['en_domain'].unique():
        # print(domain)
        genres = list(dom_genre[dom_genre['en_domain']==domain]['X-GENRE'])
        if balance: 
            if satisfy_all_genre_counts(test_target_cnt, test_curr_cnt, genres, dom_genre[dom_genre['en_domain'] == domain]):
                test_curr_cnt = update_genre_counts(test_curr_cnt, genres, dom_genre[dom_genre['en_domain'] == domain])
                test_domains.append(domain)
            elif satisfy_all_genre_counts(dev_target_cnt, dev_curr_cnt, genres, dom_genre[dom_genre['en_domain'] == domain]):
                dev_curr_cnt = update_genre_counts(dev_curr_cnt, genres, dom_genre[dom_genre['en_domain'] == domain])
                dev_domains.append(domain)
            else:
                train_domains.append(domain)
        else:
            # Add domain to test set if it satisfies the minimum target counts for all genres and doesn't go over  3 * target count for any genre
            if valid_addition(test_target_cnt, test_curr_cnt, genres, dom_genre[dom_genre['en_domain'] == domain]):
                test_curr_cnt = update_genre_counts(test_curr_cnt, genres, dom_genre[dom_genre['en_domain'] == domain])
                test_domains.append(domain)
            elif valid_addition(dev_target_cnt, dev_curr_cnt, genres, dom_genre[dom_genre['en_domain'] == domain]):
                dev_curr_cnt = update_genre_counts(dev_curr_cnt, genres, dom_genre[dom_genre['en_domain'] == domain])
                dev_domains.append(domain)
            else:
                train_domains.append(domain)
                         
    check_non_zero(test_curr_cnt, dev_curr_cnt)

    test = data[data['en_domain'].isin(test_domains)]
    dev = data[data['en_domain'].isin(dev_domains)]
    train = data[data['en_domain'].isin(train_domains)]

    return train, dev, test

def save_datasets(train, dev, test, tgt_lang, tgt_col, path, name):
    """
    Saves the datasets to tsv files with and without genre tokens, also saves complete dataframe where all 
    datasets are merged and 'set' column shows the set of each row.

    Args:
        train (pd.DataFrame): train dataset
        dev (pd.DataFrame): dev dataset
        test (pd.DataFrame): test dataset
        tgt_lang (str): target language
        path (str): path to save the files
        name (str): name of the files

    """
    
    # if tgt_col == 'par':
    # save all columns to tsv if it doesn't exist 
    if not os.path.exists(path + "/" +  name + '_complete.tsv'):
        # merge columns add test, dev or train to new column 'set'
        train['set'] = ['train'] * train.shape[0]
        dev['set'] = ['dev'] * dev.shape[0]
        test['set'] = ['test'] * test.shape[0]
        # merge all datasets
        df = pd.concat([train, dev, test])
        df.to_csv(path  + "/" +  name + '_complete.tsv', sep='\t', index=False)
    # save only en_par and is_par columns to csv

    train[[f'en_{tgt_col}', f'{tgt_lang}_{tgt_col}']].to_csv(path  + "/" +  name + '.train.tsv', sep='\t', index=False, header=False)
    dev[[f'en_{tgt_col}', f'{tgt_lang}_{tgt_col}']].to_csv(path  + "/" + name + '.dev.tsv', sep='\t', index=False, header=False)
    test[[f'en_{tgt_col}', f'{tgt_lang}_{tgt_col}']].to_csv(path  + "/" +  name + '.test.tsv', sep='\t', index=False, header=False)
    
    # add token in from of en_par according to mapping
    # genre_tokens = {'Prose/Lyrical': '>>lit<<','Instruction': '>>instr<<', 'Promotion': '>>promo<<', 'Opinion/Argumentation': '>>arg<<' , 'Other': '>>other<<' , 'Information/Explanation': '>>info<<', 'News': '>>news<<', 'Legal': '>>law<<', 'Forum': 
                    # '>>forum<<'}
    genre_tokens = {'Prose/Lyrical': '<lit>','Instruction': '<instr>', 'Promotion': '<promo>', 'Opinion/Argumentation': '<arg>' , 'Other': '<other>' , 'Information/Explanation': '<info>', 'News': '<news>', 'Legal': '<law>', 'Forum': '<forum>'}

    # make column "tokens" with genre tokens 
    train['tokens'] = train['X-GENRE'].replace(genre_tokens)
    dev['tokens'] = dev['X-GENRE'].replace(genre_tokens)
    test['tokens'] = test['X-GENRE'].replace(genre_tokens)

    # merge genre tokens with en_par in new column en_par_tokens as string
    train[f'en_{tgt_col}_tokens'] = train['tokens'] + ' ' + train[f'en_{tgt_col}']
    dev[f'en_{tgt_col}_tokens'] = dev['tokens'] + ' ' + dev[f'en_{tgt_col}']
    test[f'en_{tgt_col}_tokens'] = test['tokens'] + ' ' + test[f'en_{tgt_col}']


    # save en_par and tgt_lang_par with genre tokens

    train[[f'en_{tgt_col}_tokens', f'{tgt_lang}_{tgt_col}']].to_csv(path  + "/" +  name + '.train.tag.tsv', sep='\t', index=False, header=False)
    dev[[f'en_{tgt_col}_tokens', f'{tgt_lang}_{tgt_col}']].to_csv(path  + "/" +  name + '.dev.tag.tsv', sep='\t', index=False, header=False)
    test[[f'en_{tgt_col}_tokens', f'{tgt_lang}_{tgt_col}']].to_csv(path  + "/" +  name + '.test.tag.tsv', sep='\t', index=False, header=False)

    print('Saved datasets to ' + path  + "/" +  name + '.tsv and ' + path  + "/" +  name + '.tag.tsv and ' + path  + "/" +  name + '_complete.tsv')

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lang', '--lang_code', type=str, required=True, help='Language code of the target language')
    parser.add_argument('-len', '--length_threshold', type=int, default=25, help='Minimum length of the documents used for genre classification')
    parser.add_argument('-df', "--data_folder", type=str, default='data/', help='Folder where the data is stored')
    parser.add_argument('-download_corpus', "--download_corpus", type=bool, default=False, help='Whether to download the corpus or not')
    parser.add_argument('-preprocess', "--preprocess", type=bool, default=False, help='Whether to preprocess the corpus or not')
    parser.add_argument('-label', "--label", type=bool, default=False, help='Whether to label the corpus or not')
    parser.add_argument('-test_size', "--test_size", type=int, default=5000, help='Number of sentences to put in the test set')
    parser.add_argument('-dev_size', "--dev_size", type=int, default=5000, help='Number of sentences to put in the dev set')
    parser.add_argument('-url', '--url', type = str, help='Specify the url to download the corpus from, overrides the deafult urls in the get_url function')
    parser.add_argument('-tmx_to_json', '--tmx_to_json', type = bool, default = False, help='Whether to convert the tmx file to json or not')
    args = parser.parse_args()
    return args

def get_url(lang_code):
    if lang_code == "hr":
        url = 'https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1814/MaCoCu-hr-en.tmx.gz'
    elif lang_code == 'is':
        url = 'https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1812/MaCoCu-is-en.tmx.gz'
    return url


def main():
    args = create_arg_parser()
    data_folder = Path(args.data_folder)
    # if args.download_corpus or not Path(args.data_folder/f'MaCoCu-{args.lang_code}-en.tmx.gz').exists():
    #     url = args.url if args.url else get_url(args.lang_code)
    #     print(f"Downloading corpus from {url} and saving it as {args.data_folder/f'MaCoCu-{args.lang_code}-en.tmx.gz'}")
    #     download_corpus(url, Path(args.data_folder/f'MaCoCu-{args.lang_code}-en.tmx.gz'))
    
    if args.tmx_to_json or not Path(data_folder/f'MaCoCu-{args.lang_code}-en.json').exists():    
        tmx_to_json(data_folder/f'MaCoCu-{args.lang_code}-en.tmx', args.lang_code, data_folder/f'MaCoCu-{args.lang_code}-en.json')
    
    if args.preprocess or not Path(data_folder/f'Macocu-{args.lang_code}-en-doc-format.csv').exists():    
        print("Preprocessing started.")
        preprocess(data_folder, args.lang_code, 1, drop_par_duplicates = True, drop_doc_duplicates = False, keep_columns=True, info = False)
        print("Preprocessing done.")

    if args.label or not Path(data_folder/f'Macocu-{args.lang_code}-en-sent-doc-labelled.csv').exists():
        # load the preprocessed data
        # data= pd.read_csv(data_folder/f"Macocu-{args.lang_code}-en-doc-format-duplicates.csv", sep="\t", header=0)
        # # only use docs with length >= args.length_threshold
        # data = data[data['en_length'] >= args.length_threshold]
        # # only use unique docs for labelling to save time
        # data = data.drop_duplicates("en_doc")
        # print("Labelling started. Using docs with length >= {}".format(args.length_threshold))
        # doc_labels = classify_dataset(data, "en_doc", data_folder/f'Macocu-{args.lang_code}-en.labelled.{args.length_threshold}.csv')
        # print(f"Labelling done. Saving the labelled data to {args.data_folder}/Macocu-{args.lang_code}-en.doc.labels.{args.length_threshold}.csv")
        # # Combine the sentence level data and doc_labels
        # print(f"Combining the sentence level data and doc_labels. Saving the combined data to {args.data_folder}/Macocu-{args.lang_code}-en-sent-doc-labelled.csv")
        # # load the full dataset again to get all all sentences back 
        # del data
        # data= pd.read_csv(data_folder/f"Macocu-{args.lang_code}-en-doc-format-duplicates.csv", sep="\t", header=0)
        # # merge doc_data and data based on en_doc
        # data = pd.merge(doc_labels, data, on="en_doc")
        # load and merge all Macocu-hr-en.labelled.25.csv_*.* csv files
        doc_data = pd.DataFrame()
        for file in data_folder.glob(f"Macocu-{args.lang_code}-en.labelled.{args.length_threshold}.csv_*.*"):
            print(f"Loading {file}")
            doc_data = pd.concat([doc_data, pd.read_csv(file, sep="\t", header=0)])
        # drop all but en_doc and X-GENRE columns
        doc_data = doc_data[['en_doc', 'X-GENRE']]
        data= pd.read_csv(data_folder/f"Macocu-{args.lang_code}-en-doc-format-duplicates.csv", sep="\t", header=0)
        # merge doc_data and data based on en_doc
        data = pd.merge(doc_data, data, on="en_doc")
         # remove Unnamed: 0 column
        data = data.drop(columns=["Unnamed: 0"])
        data.to_csv(f"{args.data_folder}/Macocu-{args.lang_code}-en-sent-doc-labelled.csv", sep="\t", index=False) 
    else:
        data = pd.read_csv(data_folder/f"Macocu-{args.lang_code}-en-sent-doc-labelled.csv", sep="\t", header=0)
    
    print("Splitting the data into train, dev, test sets.")
	# make train, dev, test sets
    train, dev, test = split_data(data,test_size=args.test_size, dev_size=args.dev_size, balance = False)
    save_datasets(train, dev, test, args.lang_code, "par", args.data_folder, f"MaCoCu.en-{args.lang_code}")
    save_datasets(train.drop_duplicates(['en_doc']), dev.drop_duplicates(['en_doc']), test.drop_duplicates(['en_doc']), args.lang_code, "doc", args.data_folder, f"MaCoCu.en-{args.lang_code}.doc")
    

if __name__ == "__main__":
	main()