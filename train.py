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
                ]
    combined = f.FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()
    steps = [("extract_features", features),
             # ("classify", MultinomialNB())]
             ("classify", LinearSVC(loss='hinge'))]
             # ("classify", RandomForestRegressor(n_estimators=50, n_jobs=3, min_samples_split=10, random_state=1))]
    return Pipeline(steps)


def crossvalidate(data, num_fold):
    train = data['train']
    target = data['target']

    print '### length of data:', len(train), len(target)

    assert(len(train) == len(target))

    kf = KFold(len(train), n_folds=num_fold, shuffle=True, random_state=None)

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
    print("Reading in the training data")
    train = data_io.read_train_pairs()
    target = data_io.read_train_target()

    data = {}
    data['train'] = train
    data['target'] = target

    crossvalidate(data, 10)

    # print("Extracting features and training model")
    # classifier = get_pipeline()
    # classifier.fit(train, target.Target)
    #
    # print("Saving the classifier")
    # data_io.save_model(classifier)
    
if __name__=="__main__":
    main()
