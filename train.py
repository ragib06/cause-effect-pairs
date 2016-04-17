import data_io
import features as f
import score as sc
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import sys, getopt

def feature_extractor():
    features = [
                ('Number of Samples', 'A', f.SimpleTransform(transformer=len)),
                ('A: Number of Unique Samples', 'A', f.SimpleTransform(transformer=f.count_unique)),
                ('B: Number of Unique Samples', 'B', f.SimpleTransform(transformer=f.count_unique)),
                ('A: Normalized Entropy', 'A', f.SimpleTransform(transformer=f.normalized_entropy)),
                ('B: Normalized Entropy', 'B', f.SimpleTransform(transformer=f.normalized_entropy)),
                ('Pearson R', ['A','B'], f.MultiColumnTransform(f.correlation)),
                ('Pearson R Magnitude', ['A','B'], f.MultiColumnTransform(f.correlation_magnitude)),
                ('Entropy Difference', ['A','B'], f.MultiColumnTransform(f.entropy_difference)),
                ('Mutual Information', ['A', 'B'], f.MultiColumnTransform(transformer=f.normalized_mutual_information))
                ]
    combined = f.FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()
    steps = [("extract_features", features),
             # ("classify", MultinomialNB())]
             # ("classify", LinearSVC(loss='hinge'))]
             ("classify", RandomForestRegressor(n_estimators=50, n_jobs=3, min_samples_split=10, random_state=1))]
    return Pipeline(steps)


def crossvalidate(data, num_fold):
    train = data['train']
    target = data['target']

    print '### length of data:', len(train), len(target)

    assert(len(train) == len(target))

    kf = StratifiedKFold(target.Target, n_folds=num_fold, shuffle=True, random_state=None)
    # kf = KFold(len(train), n_folds=num_fold, shuffle=True, random_state=None)

    avg_score = 0.0

    count = 1
    for train_index, test_index in kf:
        classifier = get_pipeline()

        text_fit = classifier.fit(train.iloc[train_index, :], target.iloc[train_index, :].Target)
        predictions = text_fit.predict(train.iloc[test_index, :])
        predictions = predictions.flatten()

        score = sc.bidirectional_auc(target.iloc[test_index, :].Target, predictions)
        print '### Score(' + str(count) + '): ', score

        avg_score += score
        count += 1

    avg_score /= num_fold

    print '\n\n ******** average score:', avg_score, '********\n\n'



def main():

    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:c:h")
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(2)

    numRows = None
    cv = False
    nfold = 10

    clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
    for o, a in opts:
        if o == "-n":
            numRows = clamp(int(a), 1, 4050)
        elif o == "-c":
            cv = True
            nfold = int(a)
        elif o == "-h":
            print "python train.py -n [number of rows]"
            print "python train.py -c [number of folds]"
        else:
            print "try help: python train.py -h"
            pass


    print("Reading in the training data")
    train = data_io.read_train_pairs(numRows)
    target = data_io.read_train_target(numRows)

    if cv:
        data = {}
        data['train'] = train
        data['target'] = target

        print "Initiating " + str(nfold) + " fold cross validation ..."
        crossvalidate(data, nfold)
    else:
        print("Extracting features and training model")
        classifier = get_pipeline()
        classifier.fit(train, target.Target)

        print("Saving the classifier")
        data_io.save_model(classifier)
    
if __name__=="__main__":
    main()
