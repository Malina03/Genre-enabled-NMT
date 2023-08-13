import gzip
import shutil
import wget
import regex as re
import pandas as pd
import numpy as np
import json



def download_corpus(url):
    
    # Downloading the file by sending the request to the URL
    corpus_file = wget.download(url)
    print('Downloading Completed')

    # Unzip the file
    with gzip.open('MaCoCu-is-en.tmx.gz', 'rb') as f_in:
        with open('MaCoCu-is-en.tmx', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print('Unzipping Completed')
    return 

def tmx_to_json(corpus, tgt_language, save_path):
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
    sl_source_re = re.compile(f'<tuv xml:lang="{tgt_language}">.*?<prop type="source-document">(.*?)</prop>', re.DOTALL)
    sl_par_id_re = re.compile(f'<tuv xml:lang="{tgt_language}">.*?<prop type="paragraph-id">(.*?)</prop', re.DOTALL)
    sl_par_re = re.compile(f'<tuv xml:lang="{tgt_language}">.*?<seg>(.*?)</seg>', re.DOTALL)
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
        current_tu = {"score_bicleaner_ai": float(bi_score), "translation_direction": translation_dir, "en_source": en_source, "en_par_id": en_par_id, "en_par": en_par, "en_var_doc": en_var_doc, "en_var_dom": en_var_dom, "is_source": sl_source, "is_par_id": sl_par_id, "is_par": sl_par}
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

	# Add column for domains that are different
	corpus_df["different_domains"] = corpus_df["en_domain"] + " " + corpus_df[f"{lang_code}_domain"]

	if info:
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

	if info:
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

	if info:
		print("\nInitial length of texts in the corpus:")
		print(corpus_df.en_length.describe().to_markdown())

	# Discard instances that have length less than  79 (median from other datasets)
	corpus_df = corpus_df[corpus_df["en_length"] > length_threshold - 1]

	if info:
		print(f"\nTexts that have less than {length_threshold} words were discarded.\n")

		sentences_after_length, texts_after_length = calculate_discarded(sentences_after_text_deduplication, texts_after_text_deduplication, True)

	corpus_df['length_diff'] = abs(corpus_df[f'{lang_code}_length'] - corpus_df['en_length'])

	if info:
		# Difference in length between documents
		print("\nDifference in length between documents:\n")
		print(corpus_df['length_diff'].describe().to_markdown())

	
	if keep_columns == False:
		corpus_df = corpus_df.drop(columns = ['score_bicleaner_ai', 'en_par_id', 'en_par', f'{lang_code}_par_id', f'{lang_code}_par',  'same_domains', 'different_domains'])
		if drop_par_duplicates == True:
			corpus_df = corpus_df.drop(columns = ['en-par-src-text'])

	if info:
		# View the final dataframe
		print("The final dataframe: \n")
		print(corpus_df.head(5))
		
	if not drop_doc_duplicates:
		corpus_df.to_csv(path/f"Macocu-{lang_code}-en-doc-format-duplicates.csv", sep= "\t")
	else:
		corpus_df.to_csv(path/f"Macocu-{lang_code}-en-doc-format.csv", sep= "\t")	

	return 


def analyze_prepared_corpus(lang_code):
	"""
	Takes the CSV file, created by the filter_non_textual function and analyzes the corpus.

	Args:
	- file_name (str): path to the CSV file
	"""
	corpus_df = pd.read_csv(f"Macocu-{lang_code}-en-doc-format-filtered.csv", sep= "\t", index_col = 0)

	print("View the corpus:")
	print(corpus_df.head(3))

	# Inspect corpus information
	print("All information about the corpus: \n")
	print(corpus_df.describe(include="all"))

	# Inspect en_var_doc statistics

	print("\nPrediction of English varieties (on document level):\n")
	print(corpus_df.en_var_doc.value_counts(normalize = True).to_markdown())

	print("\nPrediction of English varieties (on domain level):\n")
	print(corpus_df.en_var_dom.value_counts(normalize = True).to_markdown())

	# Inspect translation direction
	print("\nPrediction of translation direction:\n")
	print(corpus_df.translation_direction.value_counts(normalize = True).to_markdown())

	print("\nInformation on the bicleaner score:\n")
	print(corpus_df.average_score.describe().to_markdown())

	print("\nFinal length of texts in the corpus:")
	print(corpus_df.en_length.describe().to_markdown())
	
	# Analyze English domains in the corpus_df
	count = pd.DataFrame({"Count": list(corpus_df.en_domain.value_counts())[:30], "Percentage": list(corpus_df.en_domain.value_counts(normalize="True")*100)[:30]}, index = corpus_df.en_domain.value_counts()[:30].index)

	print("\nAn analysis of the 30 most frequent English domains:")
	print(count.to_markdown())

	print("\n\nAnalysis completed.")