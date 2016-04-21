import data_io
import features as f
import score as sc
from classifier_factory import ClassifierFactory

from sklearn.pipeline import Pipeline

from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

import sys, getopt


cf = ClassifierFactory()

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
                ('Mutual Information', ['A', 'B'], f.MultiColumnTransform(transformer=f.normalized_mutual_information)),
                ('Chi-square stats stats', ['A','B'], f.MultiColumnTransform(transformer=f.chi_square_stat)),
                # ('ANOVA f_oneway stats stats', ['A','B'], f.MultiColumnTransform(transformer=f.anova_f_oneway_stat)),
                ('ANOVA kruskal stats stats', ['A','B'], f.MultiColumnTransform(transformer=f.anova_kruskal_stat))
                # ('NN_Feature1', ['A', 'B', 'A type', 'B type'], f.MultiColumnTransform(transformer=f.nn_braycurtis))
                ]

    cnt = 0
    for feat in f.NN_FEATURES:
        features.append(('NN_Feature' + str(cnt), ['A', 'B', 'A type', 'B type'], f.MultiColumnTransform(transformer=feat)))
        cnt += 1

    cnt = 0
    for feat in f.BB_FEATURES:
        features.append(('BB_Feature' + str(cnt), ['A', 'B', 'A type', 'B type'], f.MultiColumnTransform(transformer=feat)))
        cnt += 1

    combined = f.FeatureMapper(features)
    return combined


def get_pipeline(classifier_key):
    features = feature_extractor()
    steps = [("extract_features", features),
             ("classify", cf.get_classifier_obj(classifier_key))]

    return Pipeline(steps)


def crossvalidate(data, num_fold, clf_key):
    train = data['train']
    target = data['target']

    print '### length of data:', len(train), len(target)

    assert(len(train) == len(target))

    kf = StratifiedKFold(target.Target, n_folds=num_fold, shuffle=True, random_state=None)
    # kf = KFold(len(train), n_folds=num_fold, shuffle=True, random_state=None)

    avg_score = 0.0

    count = 1
    for train_index, test_index in kf:
        classifier = get_pipeline(clf_key)

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
    global cf

    numRows = None
    cv = False
    nfold = 10
    clf_keys = ["rfg"]

    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:c:m:h")
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(2)

    clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
    for o, a in opts:
        if o == "-n":
            numRows = clamp(int(a), 1, 4050)
        elif o == "-c":
            cv = True
            nfold = int(a)
        elif o == "-m":
            if a == "all":
                clf_keys = []
                for clf_key in cf.get_all_keys():
                    clf_keys.append(clf_key)
            elif cf.is_valid_key(a):
                clf_keys = [a]
            else:
                print "ERROR: wrong classifier name: " + a
        elif o == "-h":
            print 'options:'
            print "\t -n [number of rows]"
            print "\t -c [number of folds]"
            print "\t -m [classifier key | all]"
            sys.exit(0)
        else:
            print "try help: python train.py -h"
            sys.exit(2)

    print("Reading in the training data")
    train = data_io.read_train_pairs(numRows)
    trainInfo = data_io.read_train_info(numRows)
    train['A type'] = trainInfo['A type']
    train['B type'] = trainInfo['B type']
    target = data_io.read_train_target(numRows)

    if cv:
        data = {}
        data['train'] = train
        data['target'] = target

        for clf_key in clf_keys:
            print "Initiating " + str(nfold) + " fold cross validation with classifier " + cf.get_classifier_name(clf_key)
            crossvalidate(data, nfold, clf_key)
    else:
        for clf_key in clf_keys:
            print("Extracting features and training model")
            classifier = get_pipeline(clf_key)
            classifier.fit(train, target.Target)

            print("Saving the classifier")
            data_io.save_model(classifier, clf_key)
    
if __name__=="__main__":
    main()
