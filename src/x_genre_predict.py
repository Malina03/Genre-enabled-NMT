# For genre prediction on a dataset: predict labels to the texts in batches (to make the prediction faster).
# Import and install the necessary libraries
from numba import cuda
from itertools import islice
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Install transformers
# !pip install -q transformers

# Install the simpletransformers
# !pip install -q simpletransformers

from simpletransformers.classification import ClassificationModel

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lang', '--lang_code', type=str, required=True, help='Language code of the target language')
    parser.add_argument('-len', '--length_threshold', type=int, default=25, help='Minimum length of the documents used for genre classification')
    parser.add_argument('-df', "--data_folder", type=str, default='data/', help='Folder where the data is stored')
    args = parser.parse_args()
    return args



def predict(model, dataframe, final_file, dataframe_column="en_doc", softmax=False, batch_saves=1000):
    """
        The function takes the dataframe with text in column dataframe_column, creates batches of 8,
        and applies genre predictions on batches, for faster prediction.
        It saves the file with text and predictions with the final_file name.

        Args:
        - model: the model to use for prediction
        - dataframe (pandas Dataframe): specify the dataframe
        - dataframe_column (str): specify which column in the dataframe has texts to which you want to predict genres, e.g. ("docs")
        - final_file: the name of the final file with predictions

    """
    labels = ["Other", "Information/Explanation", "News", "Instruction", "Opinion/Argumentation", "Forum", "Prose/Lyrical", "Legal", "Promotion"]

        # Split the dataframe into batches
        # Create batches of text
    def chunk(arr_range, arr_size):
        arr_range = iter(arr_range)
        return iter(lambda: tuple(islice(arr_range, arr_size)), ())
    

    batches_list = list(chunk(dataframe[dataframe_column], 8))

    batches_list_new = []

    for i in batches_list:
        batches_list_new.append(list(i))

    print("The dataset is split into {} batches of {} texts.".format(len(batches_list_new),len(batches_list_new[0])))

    y_pred = []
    y_distr = []
    most_probable = []


    print("Prediction started.")
    start_time = time.time()

    batches = len(batches_list_new)
    curr_batch = 0


    for i in batches_list_new:
        if curr_batch % batch_saves == 0 and curr_batch != 0:
            print("Predicting batch {} out of {}.".format(curr_batch, batches))
            # save the dataframe with predictions every 1000 batches
            # copy first current batch to a new dataframe
            dat = dataframe.iloc[(curr_batch-batch_saves)*8:curr_batch*8]
            dat["X-GENRE"] = y_pred[(curr_batch-batch_saves)*8:curr_batch*8]
            if softmax == True:
                dat["label_distribution"] = y_distr[(curr_batch-batch_saves)*8:curr_batch*8]
                dat["chosen_category_distr"] = most_probable[(curr_batch-batch_saves)*8:curr_batch*8]
            dat.to_csv("{}_{}".format(final_file, curr_batch/batch_saves), sep="\t")
            # del dat
            return dat

        output = model.predict(i)
        current_y_pred = output[0]
        current_y_distr = output[1]
        current_y_distr_softmax = []
        current_y_distr_most_probable = []

                
        for i in current_y_pred:
            y_pred.append(i)
        
        if softmax == True:
            for i in current_y_distr:
                distr = softmax(i)
                distr_dict = {labels[i]: round(distr[i],4) for i in range(len(labels))}
                current_y_distr_softmax.append(distr_dict)
                # Also add the information for the softmax of the most probable category ("certainty")
                distr_sorted = np.sort(distr)
                current_y_distr_most_probable.append(distr_sorted[-1])

            
            for i in current_y_distr_softmax:
                y_distr.append(i)
            
            for i in current_y_distr_most_probable:
                most_probable.append(i)
        
        curr_batch += 1

    # save the final batch of predictions
    dat = dataframe.iloc[(curr_batch-batch_saves)*8:((curr_batch-batch_saves)*8+len(current_y_pred))]
    dat["X-GENRE"] = y_pred[(curr_batch-batch_saves)*8:((curr_batch-batch_saves)*8+len(current_y_pred))]
    if softmax == True:
        dat["label_distribution"] = y_distr[(curr_batch-batch_saves)*8:((curr_batch-batch_saves)*8+len(current_y_pred))]
        dat["mos_probable"] = most_probable[(curr_batch-batch_saves)*8:((curr_batch-batch_saves)*8+len(current_y_pred))]
    dat.to_csv("{}_{}".format(final_file, int(curr_batch/batch_saves)), sep="\t")
    del dat

    prediction_time = round((time.time() - start_time)/60,2)

    print("\n\nPrediction completed. It took {} minutes for {} instances - {} minutes per one instance.".format(prediction_time, dataframe.shape[0], prediction_time/dataframe.shape[0]))

    # load the saved predictions and add them to the dataframe
    for i in range(1, int(curr_batch/batch_saves)):
        if i == 1:
            res = pd.read_csv("{}_{}".format(final_file, i), sep="\t")
        else:
            res = res.append(pd.read_csv("{}_{}".format(final_file, i), sep="\t"))

    res.to_csv("{}".format(final_file), sep="\t")

    return res

def classify_dataset(data, tgt_column, save_file, softmax=False):
    """
        The function takes the dataframe with text in column dataframe_column, creates batches of 8,
        and applies genre predictions on batches, for faster prediction.
        It saves the file with text and predictions with the final_file name.
    
        
        Args:
        - dataframe (pandas Dataframe): specify the dataframe
        - tgt_column (str): specify which column in the dataframe has texts to which you want to predict genres, e.g. ("en_docs")
        - save_file (str): the name of the final file with predictions

        Returns:
        - dataframe (pandas Dataframe): the dataframe with predictions to integrate it in the preprocessing pipeline

    """
    model_args= {
                "num_train_epochs": 15,
                "learning_rate": 1e-5,
                "max_seq_length": 512,
                "silent": True
                }

    model = ClassificationModel(
        "xlmroberta",
        "classla/xlm-roberta-base-multilingual-text-genre-classifier",
        use_cuda=True,
        args=model_args
    )

    labelled = predict(model, data, save_file, tgt_column, softmax=softmax, batch_saves=5)
    # remove all columns except en_doc and X-GENRE to save memory
    if softmax == False:
        labelled = labelled[["en_doc", "X-GENRE"]]
    else:
        labelled = labelled[["en_doc", "X-GENRE", "label_distribution", "chosen_category_distr"]]
    return labelled

    # Apply softmax to the raw outputs
def softmax(x):
    '''Compute softmax values for each sets of scores in x.'''
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def main():
    '''Makes predictions on the dataset returning the full output, label distribution and most probable label.'''
    args = create_arg_parser()
    data_folder = Path(args.data_folder)
    # Load the dataframe
    data= pd.read_csv(data_folder/f"Macocu-{args.lang_code}-en-doc-format-duplicates.csv", sep="\t", header=0)
    # only use docs with length >= args.length_threshold
    data = data[data['en_length'] >= args.length_threshold]
    # only use unique docs for labelling to save time
    data = data.drop_duplicates("en_doc")
    # only save the en_doc column to save memory
    data = data[["en_doc"]]
    print("Labelling started. Using docs with length >= {}".format(args.length_threshold))
    doc_labels = classify_dataset(data, "en_doc", data_folder/f'Macocu-{args.lang_code}-en.labelled.softmax{args.length_threshold}.csv', softmax=True)
    print(f"Labelling done. Saving the labelled data to {args.data_folder}/Macocu-{args.lang_code}-en.doc.labels.softmax.{args.length_threshold}.csv")
    # Combine the sentence level data and doc_labels
    print(f"Combining the sentence level data and doc_labels. Saving the combined data to {args.data_folder}/Macocu-{args.lang_code}-en-sent-doc-labelled-softmax.csv")
    # load the full dataset again to get all all sentences back 
    del data
    data= pd.read_csv(data_folder/f"Macocu-{args.lang_code}-en-doc-format-duplicates.csv", sep="\t", header=0)
    # merge doc_data and data based on en_doc
    data = pd.merge(doc_labels, data, on="en_doc")
    data = data.drop(columns=["Unnamed: 0"])
    data.to_csv(f"{args.data_folder}/Macocu-{args.lang_code}-en-sent-doc-labelled-softmax.csv", sep="\t", index=False) 


if __name__ == "__main__":
    main()