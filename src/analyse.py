import pandas as pd
# import matplotlib.pyplot as plt

def print_genre_distribution(data):
    print("Genre distribution:")
    print(data['X-GENRE'].value_counts())
    print("Total number of lines:", len(data))

# def plot_genre_distribution(data, save_file):
#     data['X-GENRE'].value_counts().plot(kind='bar')
#     plt.savefig(save_file)

def main():
    data = pd.read_csv('/scratch/s3412768/genre_NMT/en-hr/data/MaCoCu.en-hr_complete.tsv', sep='\t')
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

if __name__ == "__main__":
    main()