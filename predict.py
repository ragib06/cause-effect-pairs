import data_io
import numpy as np
import pickle
import sys, getopt
from classifier_factory import ClassifierFactory

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
    cf = ClassifierFactory()

    filename = None
    modelnames = ["basic_python_benchmark"]
    numRows = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:m:n:h")
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(2)

    for o, a in opts:
        if o == "-f":
            filename = a
        elif o == "-n":
            numRows = int(a)
        elif o == "-m":
            if a == "all":
                modelnames = []
                for clf_key in cf.get_all_keys():
                    modelnames.append(clf_key)
            elif cf.is_valid_key(a):
                modelnames = [a]
        elif o == "-h":
            print 'options:'
            print "\t -m [classifier key | all]"
            print "\t -f [filename]"
            sys.exit(0)
        else:
            print "try help: python predict.py -h"
            sys.exit(1)

    print "Reading the test pairs"
    test = data_io.read_test_pairs(numRows)
    testInfo = data_io.read_test_info(numRows)
    test['A type'] = testInfo['A type']
    test['B type'] = testInfo['B type']

    for modelname in modelnames:
        print "Loading the classifier:", cf.get_classifier_name(modelname)
        classifier = data_io.load_model(modelname)

        print "Making predictions"
        predictions = classifier.predict(test)
        predictions = predictions.flatten()

        filename = modelname + '.csv'

        data_io.write_submission(predictions, filename)

if __name__=="__main__":
    main()