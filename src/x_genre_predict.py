# For genre prediction on a dataset: predict labels to the texts in batches (to make the prediction faster).
# Import and install the necessary libraries
from numba import cuda
from itertools import islice
import time
from tqdm import tqdm
import pandas as pd

# Install transformers
# !pip install -q transformers

# Install the simpletransformers
# !pip install -q simpletransformers

from simpletransformers.classification import ClassificationModel


def predict(model, dataframe, final_file, dataframe_column="en_doc"):
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

    print("Prediction started.")
    start_time = time.time()

    batches = len(batches_list_new)
    curr_batch = 0

    ## added to finish timedout run
    batches_list_new = batches_list_new[0:1000*16]
    curr_batch = 1000*16

    for i in batches_list_new:
        if curr_batch % 1000 == 0:
            print("Predicting batch {} out of {}.".format(curr_batch, batches))
            # save the dataframe with predictions every 1000 batches
            # copy first current batch to a new dataframe
            dat = dataframe.iloc[(curr_batch-1000)*8:curr_batch*8]
            dat["X-GENRE"] = y_pred[(curr_batch-1000)*8:curr_batch*8]
            dat.to_csv("{}_{}".format(final_file, curr_batch/1000), sep="\t")
            del dat
        
        output = model.predict(i)
        current_y_pred = [model.config.id2label[i] for i in output[0]]

        for i in current_y_pred:
            y_pred.append(i)
        curr_batch += 1

    # save the final batch of predictions
    dat = dataframe.iloc[(curr_batch-1000)*8:curr_batch*8]
    dat["X-GENRE"] = y_pred[(curr_batch-1000)*8:curr_batch*8]
    dat.to_csv("{}_{}".format(final_file, curr_batch/1000+1), sep="\t")
    del dat

    prediction_time = round((time.time() - start_time)/60,2)

    print("\n\nPrediction completed. It took {} minutes for {} instances - {} minutes per one instance.".format(prediction_time, dataframe.shape[0], prediction_time/dataframe.shape[0]))

    # load the saved predictions and add them to the dataframe
    for i in range(1, curr_batch/1000 + 1):
        if i == 1:
            res = pd.read_csv("{}_{}".format(final_file, i), sep="\t")
        else:
            res = res.append(pd.read_csv("{}_{}".format(final_file, i), sep="\t"))

    res.to_csv("{}".format(final_file), sep="\t")

    return res


    # # dataframe["X-GENRE"] = y_pred

    # # Save the new dataframe which contains the y_pred values as well
    # dataframe.to_csv("{}".format(final_file), sep="\t")

    # return dataframe


def classify_dataset(data, tgt_column, save_file):
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

    labelled = predict(model, data, save_file, tgt_column)
    # remove all columns except en_doc and X-GENRE to save memory
    labelled = labelled[["en_doc", "X-GENRE"]]
    return labelled

