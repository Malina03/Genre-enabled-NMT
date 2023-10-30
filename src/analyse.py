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
    labels = ["Other", "Information/Explanation", "News", "Instruction", "Opinion/Argumentation", "Forum", "Prose/Lyrical", "Legal", "Promotion"]
    data = pd.read_csv('/scratch/s3412768/genre_NMT/en-hr/data/MaCoCu.en-hr_complete.tsv', sep='\t')
    labels_distr = pd.read_csv('/scratch/s3412768/genre_NMT/en-hr/data/softmax_saves/Macocu-hr-en-sent-doc-labelled-softmax.csv', sep='\t')
    # remove all but en_par, X-GENRE, "label_distribution", "chosen_category_distr"
    labels_distr = labels_distr[['en_par', 'X-GENRE', 'label_distribution', 'chosen_category_distr']]
    # rename X-GENRE to X_GENRE_softmax
    labels_distr = labels_distr.rename(columns={'X-GENRE': 'X-GENRE_softmax'})
    data = pd.merge(data, labels_distr, on='en_par')
    # check if X-GENRE and X-GENRE_softmax are the same
    print("Check if X-GENRE and X-GENRE_softmax are the same:")
    data['same_classification'] = data.apply(lambda x: x['X-GENRE']==x['X-GENRE_softmax'], axis=1)
    print(data['same_classification'].value_counts())
    # remove X-GENRE_softmax if the same as X-GENRE
    if data['same_classification'].value_counts()[True] == len(data):
        data = data.drop(columns=['X-GENRE_softmax'])

    # check if the sources are different by set
    dev_src = data[data['set']=='dev']['en_domain'].unique()
    test_src = data[data['set']=='test']['en_domain'].unique()
    train_src = data[data['set']=='train']['en_domain'].unique()
    print("Check if the sources are different by set")
    print(set(test_src).intersection(set(train_src)).intersection(set(dev_src)))

    print ("Number of lines in the dataset:", len(data))
    print("Gnre distribution in the entire dataset:")
    print(data['X-GENRE'].value_counts())
    print("Genre distribution in the training set:")
    print(data[data['set']=='train']['X-GENRE'].value_counts())
    print("Genre distribution in the development set:")
    print(data[data['set']=='dev']['X-GENRE'].value_counts())
    print("Genre distribution in the test set:")
    print(data[data['set']=='test']['X-GENRE'].value_counts())

    print("\n\n\n")

    # average chosen category distribution per genre
    print("Average chosen category distribution per genre:")
    print(data.groupby('X-GENRE')['chosen_category_distr'].mean())
    # median chosen category distribution per genre
    print("Median chosen category distribution per genre:")
    print(data.groupby('X-GENRE')['chosen_category_distr'].median())
    # average chosen category distribution per genre per set
    print("Average chosen category distribution per genre per set:")
    print(data.groupby(['X-GENRE', 'set'])['chosen_category_distr'].mean())
    # median chosen category distribution per genre per set
    print("Median chosen category distribution per genre per set:")
    print(data.groupby(['X-GENRE', 'set'])['chosen_category_distr'].median())

    # plot label distribution per genre
    for genre in labels:
        data[data['X-GENRE']==genre]['chosen_category_distr'].plot(kind='hist', title=genre)
        plt.savefig('/scratch/s3412768/genre_NMT/en-hr/data/softmax_saves/label_distr_plot_'+genre+'.png')

    print("\n\n\n")
    # number of labels of each genre with above 0.9 confidence
    print("Number of labels of each genre with above 0.9 confidence:")
    print(data[data['chosen_category_distr']>=0.9].groupby('X-GENRE')['chosen_category_distr'].count())
    print("Percentage of labels of each genre with above 0.95 confidence:")
    print(data[data['chosen_category_distr']>=0.9].groupby('X-GENRE')['chosen_category_distr'].count()/data.groupby('X-GENRE')['chosen_category_distr'].count())

    # add columns with the confidence of each label from the label distribution
    

    # transform label distribution from string to dictionary
    data['label_distribution'] = data['label_distribution'].apply(lambda x: json.loads(x.replace("'", "\"")))
    
    # check if the label distribution is a dictionary
    print("Check if the label distribution is a dictionary:")
    print(data['label_distribution'].apply(lambda x: type(x)==dict).value_counts())

    

    for i in range(0, len(labels)):
        data['label_'+labels[i]+'_conf'] = data['label_distribution'].apply(lambda x: float(x[labels[i]]))
    
    
    # make column for 2nd most confident category by comparing the label confidences
    data['2nd_most_confident_dist'] = data.apply(lambda x: sorted([x['label_'+labels[i]+'_conf'] for i in range(0, len(labels))], reverse=True)[1], axis=1)
    data['2nd_most_confident'] = data.apply(lambda x: labels[[x['label_'+labels[i]+'_conf'] for i in range(0, len(labels))].index(sorted([x['label_'+labels[i]+'_conf'] for i in range(0, len(labels))], reverse=True)[1])], axis=1)
    # average 2nd most confident category distribution per genre
    print("Average 2nd most confident category distribution per genre:")
    print(data.groupby('X-GENRE')['2nd_most_confident_dist'].mean())
    # median 2nd most confident category distribution per genre
    print("Median 2nd most confident category distribution per genre:")
    print(data.groupby('X-GENRE')['2nd_most_confident_dist'].median())

    # the 2nd most likely label per genre 
    print("The 2nd most likely label per genre:")
    print(data.groupby('X-GENRE')['2nd_most_confident'].value_counts())

    print("\n\n\n")

    data = data.drop_duplicates(subset=['en_doc'])
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