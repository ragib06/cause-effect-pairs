import csv
import json
import numpy as np
import os
import pandas as pd
import pickle
import features as f

def get_paths():
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def parse_dataframe(df):
    parse_cell = lambda cell: np.fromstring(cell, dtype=np.float, sep=" ")
    df = df.applymap(parse_cell)
    return df

def parse_dataframe_type(df):
    parse_cell = lambda cell: f.get_feature_data_type(cell)
    df = df.applymap(parse_cell)
    return df

def read_train_pairs(num_rows = None):
    train_path = get_paths()["train_pairs_path"]
    return parse_dataframe(pd.read_csv(train_path, index_col="SampleID", nrows=num_rows))

def read_train_target(num_rows = None):
    path = get_paths()["train_target_path"]
    df = pd.read_csv(path, index_col="SampleID", nrows = num_rows)
    df = df.rename(columns = dict(zip(df.columns, ["Target", "Details"])))
    return df

def read_train_info(num_rows = None):
    info_path = get_paths()["train_info_path"]
    return parse_dataframe_type(pd.read_csv(info_path, index_col="SampleID", nrows=num_rows))

def read_test_pairs(num_rows=None):
    valid_path = get_paths()["test_pairs_path"]
    return parse_dataframe(pd.read_csv(valid_path, index_col="SampleID", nrows=num_rows))

def read_test_info(num_rows=None):
    path = get_paths()["test_info_path"]
    return parse_dataframe_type(pd.read_csv(path, index_col="SampleID", nrows=num_rows))

def read_solution():
    solution_path = get_paths()["solution_path"]
    return pd.read_csv(solution_path, index_col="SampleID")

def save_model(model, clf_key):
    out_path = get_paths()["model_path"]
    out_path = '/'.join(out_path.split('/')[:-1]) + '/' + clf_key + '.pickle'
    print 'saving model to ' + out_path

    pickle.dump(model, open(out_path, "w"))

def load_model(modelname=None):
    in_path = get_paths()["model_path"]
    if modelname is not None:
        in_path = '/'.join(in_path.split('/')[:-1]) + '/' + modelname + '.pickle'
    return pickle.load(open(in_path))

def read_submission():
    submission_path = get_paths()["submission_path"]
    return pd.read_csv(submission_path, index_col="SampleID")

def write_submission(predictions, filename=None):
    submission_path = get_paths()["submission_path"]
    if filename is not None:
        submission_path = '/'.join(submission_path.split('/')[:-1]) + '/' + filename
    print 'writing submission to ' + submission_path
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    test = read_test_pairs()
    rows = [x for x in zip(test.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)
