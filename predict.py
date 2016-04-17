import data_io
import numpy as np
import pickle
import sys, getopt

def historic():
    print("Calculating correlations")
    calculate_pearsonr = lambda row: abs(pearsonr(row["A"], row["B"])[0])
    correlations = valid.apply(calculate_pearsonr, axis=1)
    correlations = np.array(correlations)

    print("Calculating causal relations")
    calculate_causal = lambda row: causal_relation(row["A"], row["B"])
    causal_relations = valid.apply(calculate_causal, axis=1)
    causal_relations = np.array(causal_relations)

    scores = correlations * causal_relations

def main():

    filename = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:")
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(2)

    for o, a in opts:
        if o == "-f":
            filename = a

    print("Reading the test pairs")
    valid = data_io.read_test_pairs()

    print("Loading the classifier")
    classifier = data_io.load_model()

    print("Making predictions")
    predictions = classifier.predict(valid)
    predictions = predictions.flatten()

    print("Writing predictions to file")
    data_io.write_submission(predictions, filename)

if __name__=="__main__":
    main()