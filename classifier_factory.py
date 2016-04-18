__author__ = 'ragib'

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class ClassifierFactory:

    def __init__(self):

        self.factoryMap = {
            "lr"    :   {
                "name"  :   "LogisticRegression",
                "obj"   :   LogisticRegression(random_state=0)
            },
            "lsvc"  :   {
                "name"  :   "LinearSVC",
                "obj"   :   LinearSVC(loss='hinge')
            },
            "dtc"    :   {
                "name"  :   "DecisionTreeClassifier",
                "obj"   :   DecisionTreeClassifier(random_state=0)
            },
            "rfc"   :   {
                "name"  :   "RandomForestClassifier",
                "obj"   :   RandomForestClassifier(random_state=0)
            },
            "rfg"   :   {
                "name"  :   "RandomForestRegressor",
                "obj"   :   RandomForestRegressor(n_estimators=50, n_jobs=3, min_samples_split=10, random_state=1)
            },
            "gbc"   :   {
                "name"  :   "GradientBoostingClassifier",
                "obj"   :   GradientBoostingClassifier(subsample=0.5, n_estimators=10, random_state=0)
            },
            "knc"   :   {
                "name"  :   "kNeighborsClassifier",
                "obj"   :   KNeighborsClassifier()
            },
            "gnb"   :   {
                "name"  :   "GaussianNB",
                "obj"   :   GaussianNB()
            },
        }

    def get_classifier(self, key):
        return self.factoryMap[key]

    def get_classifier_obj(self, key):
        return self.factoryMap[key]['obj']

    def get_classifier_name(self, key):
        return self.factoryMap[key]['name']

    def is_valid_key(self, key):
        return self.factoryMap.has_key(key)

    def get_all_keys(self):
        return self.factoryMap.keys()




