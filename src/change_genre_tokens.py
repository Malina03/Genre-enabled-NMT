import pandas as pd
import os


data_folder = "/scratch/s3412768/genre_NMT/en-hr/data/"

# get all files in data_folder/old_tokens

files = os.listdir(data_folder + "old_tokens/")

for f in files:
    # read in the file
    df = pd.read_csv(data_folder + "old_tokens/" + f, sep="\t", header=None, names=["source", "target"])
    # replace >> with < and << with >
    df["source"] = df["source"].str.replace(">>", "<")
    df["source"] = df["source"].str.replace("<<", ">")
    # write the file
    df.to_csv(data_folder + f, sep="\t", header=None, index=False)

