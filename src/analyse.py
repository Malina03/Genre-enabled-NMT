import pandas as pd
import matplotlib.pyplot as plt
import json 

def print_genre_distribution(data):
    print("Genre distribution:")
    print(data['X-GENRE'].value_counts())
    print("Total number of lines:", len(data))

# def plot_genre_distribution(data, save_file):
#     data['X-GENRE'].value_counts().plot(kind='bar')
#     plt.savefig(save_file)

def main():
    # labels = ["Other", "Information/Explanation", "News", "Instruction", "Opinion/Argumentation", "Forum", "Prose/Lyrical", "Legal", "Promotion"]
    # data = pd.read_csv('/scratch/s3412768/genre_NMT/en-hr/data/MaCoCu.en-is_complete.tsv', sep='\t', header=0)
    # print(data.shape())
    # only_doc = data.drop_duplicates('en_doc')
    # print(only_doc.shape())
    # # only keep set = test
    # only_doc = only_doc[only_doc['set']=='test']
    # print(only_doc.shape())
    # print("Croatian median, mean, and std of the document length:")
    # print(only_doc['er_length'].median())
    # print(only_doc['en_length'].mean())
    # print(only_doc['en_length'].std())

    # data = pd.read_csv('/scratch/s3412768/genre_NMT/en-is/data/MaCoCu.en-is_complete.tsv', sep='\t', header=0)
    # only_doc = data.drop_duplicates('en_doc')
    # # only keep set = test
    # only_doc = only_doc[only_doc['set']=='test']
    # print("Icelandic median, mean, and std of the document length:")
    # print(only_doc['er_length'].median())
    # print(only_doc['en_length'].mean())
    # print(only_doc['en_length'].std())
    
    # read json file
    # with open('/scratch/s3412768/genre_NMT/en-hr/data/softmax_saves/label_distribution.json') as f:
    #     data = json.load(f)
    #



    data = pd.read_csv('/scratch/s3412768/genre_NMT/en-hr/data/MaCoCu.en-hr.doc_complete.tsv', sep='\t', header=0)




    # dat1 = pd.read_csv('/scratch/s3412768/genre_NMT/en-tr/data/MaCoCu.en-tr.train.tag.tsv', sep='\t', header=None)
    # # name columns
    # dat2 = pd.read_csv('/scratch/s3412768/genre_NMT/en-tr/data/MaCoCu.en-tr.dev.tag.tsv', sep='\t', header=None)
    # dat3 = pd.read_csv('/scratch/s3412768/genre_NMT/en-tr/data/MaCoCu.en-tr.test.tag.tsv', sep='\t', header=None)
    # print(dat1.shape)
    # print(dat2.shape)
    # print(dat3.shape)
    # # put df on top of each other
    # data = pd.concat([dat1, dat2], axis=0)
    # data = pd.concat([data, dat3], axis=0)
    # # reset index
    # data = data.reset_index(drop=True)
    # # rename columns
    # data.columns = ['en_par', 'tr_par']
    # data['genre'] = data['en_par'].apply(lambda x: x.split(' ')[0])
    # print(data['genre'].value_counts())
    # labels_distr = pd.read_csv('/scratch/s3412768/genre_NMT/en-tr/data/Macocu-tr-en-sent-doc-labelled-softmax.csv', sep='\t')
    # # remove all but en_par, X-GENRE, "label_distribution", "chosen_category_distr"
    # labels_distr = labels_distr[['en-par-src-text', 'X-GENRE', 'label_distribution', 'chosen_category_distr']]
    # # rename X-GENRE to X_GENRE_softmax
    # labels_distr = labels_distr.rename(columns={'X-GENRE': 'X-GENRE_softmax'})
    # data = pd.merge(data, labels_distr, on='en-par-src-text')
    
    # # transform label distribution from string to dictionary
    # data['label_distribution'] = data['label_distribution'].apply(lambda x: json.loads(x.replace("'", "\"")))
    
    # # check if the label distribution is a dictionary
    # print("Check if the label distribution is a dictionary:")
    # print(data['label_distribution'].apply(lambda x: type(x)==dict).value_counts())
    # # add columns with the confidence of each label from the label distribution
    # for i in range(0, len(labels)):
    #     data['label_'+labels[i]+'_conf'] = data['label_distribution'].apply(lambda x: float(x[labels[i]]))
    # # change X-GENRE_softmax values to the most confident label
    # data['X-GENRE_softmax'] = data.apply(lambda x: labels[[x['label_'+labels[i]+'_conf'] for i in range(0, len(labels))].index(max([x['label_'+labels[i]+'_conf'] for i in range(0, len(labels))]))], axis=1)
    # # cahnge chosen_category_distr to the confidence of the chosen category
    # data['chosen_category_distr'] = data.apply(lambda x: x['label_'+x['X-GENRE_softmax']+'_conf'], axis=1)

    # # check if X-GENRE and X-GENRE_softmax are the same
    # print("Check if X-GENRE and X-GENRE_softmax are the same:")
    # data['same_classification'] = data.apply(lambda x: x['X-GENRE']==x['X-GENRE_softmax'], axis=1)
    # #print to a file en_par hr_par en_doc hr_doc X-GENRE X-GENRE_softmax same_classification of same_classification = False
    # data[data['same_classification']==False][['en_par', 'hr_par', 'en_doc', 'hr_doc', 'X-GENRE', 'X-GENRE_softmax', 'label_distribution']].to_csv('/scratch/s3412768/genre_NMT/en-hr/data/softmax_saves/different_classification.csv', sep='\t')

    # print(data['same_classification'].value_counts())
    # # remove X-GENRE_softmax if the same as X-GENRE
    # if data['same_classification'].value_counts()[True] == len(data):
    #     data = data.drop(columns=['X-GENRE_softmax'])

    # # check if the sources are different by set
    # dev_src = data[data['set']=='dev']['en_domain'].unique()
    # test_src = data[data['set']=='test']['en_domain'].unique()
    # train_src = data[data['set']=='train']['en_domain'].unique()
    # print("Check if the sources are different by set")
    # print(set(test_src).intersection(set(train_src)).intersection(set(dev_src)))

    # print ("Number of lines in the dataset:", len(data))
    # print("Gnre distribution in the entire dataset:")
    # print(data['X-GENRE'].value_counts())
    # print("Genre distribution in the training set:")
    # print(data[data['set']=='train']['X-GENRE'].value_counts())
    # print("Genre distribution in the development set:")
    # print(data[data['set']=='dev']['X-GENRE'].value_counts())
    # print("Genre distribution in the test set:")
    # print(data[data['set']=='test']['X-GENRE'].value_counts())

    # print("\n\n\n")

    # # average chosen category distribution per genre
    # print("Average chosen category distribution per genre:")
    # print(data.groupby('X-GENRE')['chosen_category_distr'].mean())
    # # median chosen category distribution per genre
    # print("Median chosen category distribution per genre:")
    # print(data.groupby('X-GENRE')['chosen_category_distr'].median())
    # # average chosen category distribution per genre per set
    # print("Average chosen category distribution per genre per set:")
    # print(data.groupby(['X-GENRE', 'set'])['chosen_category_distr'].mean())
    # # median chosen category distribution per genre per set
    # print("Median chosen category distribution per genre per set:")
    # print(data.groupby(['X-GENRE', 'set'])['chosen_category_distr'].median())

    # # plot label distribution per genre
    # for genre in labels:
    #     plt.figure()
    #     data[data['X-GENRE']==genre]['chosen_category_distr'].plot(kind='hist', title=genre)
    #     ## add average and median lines to the plot
    #     plt.axvline(data[data['X-GENRE']==genre]['chosen_category_distr'].mean(), color='m', linestyle='dashed', linewidth=1)
    #     plt.axvline(data[data['X-GENRE']==genre]['chosen_category_distr'].median(), color='r', linestyle='dashed', linewidth=1)
    #     plt.legend()
    #     plt.savefig('/scratch/s3412768/genre_NMT/en-hr/data/softmax_saves/label_distr_plot_'+genre.replace("/", "-")+'.png')

    # print("\n\n\n")
    # # number of labels of each genre with above 0.9 confidence
    # print("Number of labels of each genre with above 0.9 confidence:")
    # print(data[data['chosen_category_distr']>=0.9].groupby('X-GENRE')['chosen_category_distr'].count())
    # print("Percentage of labels of each genre with above 0.95 confidence:")
    # print(data[data['chosen_category_distr']>=0.9].groupby('X-GENRE')['chosen_category_distr'].count()/data.groupby('X-GENRE')['chosen_category_distr'].count())
    
    
    # # make column for 2nd most confident category by comparing the label confidences
    # data['2nd_most_confident_dist'] = data.apply(lambda x: sorted([x['label_'+labels[i]+'_conf'] for i in range(0, len(labels))], reverse=True)[1], axis=1)
    # data['2nd_most_confident'] = data.apply(lambda x: labels[[x['label_'+labels[i]+'_conf'] for i in range(0, len(labels))].index(sorted([x['label_'+labels[i]+'_conf'] for i in range(0, len(labels))], reverse=True)[1])], axis=1)
    # # average 2nd most confident category distribution per genre
    # print("Average 2nd most confident category distribution per genre:")
    # print(data.groupby('X-GENRE')['2nd_most_confident_dist'].mean())
    # # median 2nd most confident category distribution per genre
    # print("Median 2nd most confident category distribution per genre:")
    # print(data.groupby('X-GENRE')['2nd_most_confident_dist'].median())

    # # the 2nd most likely label per genre 
    # print("The 2nd most likely label per genre:")
    # print(data.groupby('X-GENRE')['2nd_most_confident'].value_counts())

    # # plot most likely 2nd label per genre 

    # for genre in labels:
    #     plt.figure()
    #     for label in data[data['X-GENRE']==genre]['2nd_most_confident'].unique():
    #         plt.bar(label, data[data['X-GENRE']==genre]['2nd_most_confident'].value_counts()[label], label=label)
    #         ## add 2nd_most confident_dist average as value on top of the bar
    #         plt.text(label, data[data['X-GENRE']==genre]['2nd_most_confident'].value_counts()[label], round(data[data['X-GENRE']==genre][data['2nd_most_confident']==label]['2nd_most_confident_dist'].mean(), 2), ha='center', va='bottom')
    #     # add legend
    #     plt.legend()
    #     plt.savefig('/scratch/s3412768/genre_NMT/en-hr/data/softmax_saves/2nd_most_confident_'+genre.replace("/", "-")+'.png')

    # print("\n\n\n")

    data = data.drop_duplicates(subset=['en_doc'])
    data['hr_doc_length'] = data['hr_doc'].apply(lambda x: len(x.split(' ')))
    data['en_doc_length'] = data['en_doc'].apply(lambda x: len(x.split(' ')))

    # print("Average document length in test set: {} and std {} and median {}".format(data[data['set']=='test']['tr_doc_length'].mean(), data[data['set']=='test']['tr_doc_length'].std(), data[data['set']=='test']['tr_doc_length'].median()))
    # print("Average document length in test set English: {} and std {} and median {}".format(data[data['set']=='test']['en_doc_length'].mean(), data[data['set']=='test']['en_doc_length'].std(), data[data['set']=='test']['en_doc_length'].median()))
    print("Average doc length in the test set:", data[data['set']=='test']['hr_doc_length'].mean())
    print("Standard deviation of the doc length in the test set:", data[data['set']=='test']['hr_doc_length'].std())
    print("Median doc length in the test set:", data[data['set']=='test']['hr_doc_length'].median())
    print ("Number of docs in the dataset:", len(data))
    print("Gnre distribution in the entire dataset:")
    print(data['X-GENRE'].value_counts())
    print("Genre distribution in the training set:")
    print(data[data['set']=='train']['X-GENRE'].value_counts())
    print("Genre distribution in the development set:")
    print(data[data['set']=='dev']['X-GENRE'].value_counts())
    print("Genre distribution in the test set:")
    print(data[data['set']=='test']['X-GENRE'].value_counts())

    print("\n\n\n")

if __name__ == "__main__":
    main()