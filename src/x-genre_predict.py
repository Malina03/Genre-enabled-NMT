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


def predict(model, dataframe, final_file):
    """
        The function takes the dataframe with text in column dataframe_column, creates batches of 8,
        and applies genre predictions on batches, for faster prediction.
        It saves the file with text and predictions with the final_file name.

        Args:
        - dataframe (pandas Dataframe): specify the dataframe
        - dataframe_column (str): specify which column in the dataframe has texts to which you want to predict genres, e.g. ("docs")
        - final_file: the name of the final file with predictions

    """
        # Split the dataframe into batches
        # Create batches of text
    def chunk(arr_range, arr_size):
        arr_range = iter(arr_range)
        return iter(lambda: tuple(islice(arr_range, arr_size)), ())

    batches_list = list(chunk(dataframe["en_doc"], 8))

    batches_list_new = []

    for i in batches_list:
        batches_list_new.append(list(i))

    print("The dataset is split into {} batches of {} texts.".format(len(batches_list_new),len(batches_list_new[0])))

    y_pred = []

    print("Prediction started.")
    start_time = time.time()

    for i in tqdm(batches_list_new):
        output = model.predict(i)
        current_y_pred = [model.config.id2label[i] for i in output[0]]

        for i in current_y_pred:
            y_pred.append(i)

    prediction_time = round((time.time() - start_time)/60,2)

    print("\n\nPrediction completed. It took {} minutes for {} instances - {} minutes per one instance.".format(prediction_time, dataframe.shape[0], prediction_time/dataframe.shape[0]))

    dataframe["X-GENRE"] = y_pred

    # Save the new dataframe which contains the y_pred values as well
    dataframe.to_csv("{}".format(final_file), sep="\t")

    return dataframe



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

data = pd.read_csv("data/preprocessed_50.csv", sep="\t", header=0)
predict(model, data, "data/labelled_50.csv")

data = pd.read_csv("data/preprocessed_75.csv", sep="\t", header=0)
predict(model, data, "data/labelled_75.csv")

data = pd.read_csv("data/preprocessed_150.csv", sep="\t", header=0)
predict(model, data, "data/labelled_150.csv")

# data = pd.read_csv("data/random_sample_75.csv", sep="\t", header=0)
# predict(model, data, "data/sample_75_labelled.csv")

# data = pd.read_csv("data/random_sample_50.csv", sep="\t", header=0)
# predict(model, data, "data/sample_50_labelled.csv")

# data = pd.read_csv("data/random_sample_25.csv", sep="\t", header=0)
# predict(model, data, "data/sample_25_labelled.csv")