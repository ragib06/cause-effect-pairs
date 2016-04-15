Cause Effect Pairs Challenge
============================

This repo intiated from a copy of benchmark and sample code in Python for the [Cause Effect Pairs Challenge](https://www.kaggle.com/c/cause-effect-pairs), a machine learning challenged hosted by [Kaggle](https://www.kaggle.com) and organized by [ChaLearn](http://www.chalearn.org/).

Executing this requires Python 2.7 along with the following packages:

 - pandas (tested with version 10.1)
 - sklearn (tested with version 0.13)
 - numpy (tested with version 1.6.2)
 - scipy (tested with version 0.10.)
 - ml_metrics

To run,

1. [Download the data](https://www.kaggle.com/c/cause-effect-pairs/data)
2. Create three directories inside the repo directory: data, models, submissions
3. Extract the kaggle data inside the “data” directory such that this is a valid path: data/CEfinal_train_text/CEfinal_train_pairs.csv
4. Modify SETTINGS.json to point to the training and validation data on your system, as well as a place to save the trained model and a place to save the submission
5. Now to train the classifier run "python train.py", it will save the model in models directory
6. Otherwise, to cross-validate, run "python train.py -c 10" [10 fold cv]
7. To try with a small subset of data, run "python train.py -n 100" [first 100 rows]
8. Experiment with different classifiers in get_pipeline() function in train.py
9. So, "python train.py -n 100 -c 3" means it will take first 100 rows and run a 3-fold cross-validation
10. Make predictions on the validation set by running `python predict.py` [check the path]
11. [Make a submission](https://www.kaggle.com/c/cause-effect-pairs/team/select) with the output file in submissions directory