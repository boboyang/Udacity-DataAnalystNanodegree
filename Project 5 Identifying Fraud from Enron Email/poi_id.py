#!/usr/bin/python

import pickle
import sys

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest

all_features = ['poi', 'salary', 'deferral_payments',
                'total_payments',
                'loan_advances', 'bonus',
                'restricted_stock_deferred', 'deferred_income',
                'total_stock_value', 'expenses',
                'exercised_stock_options',
                'other', 'long_term_incentive',
                'restricted_stock', 'director_fees',
                'to_messages',
                'from_poi_to_this_person', 'from_messages',
                'from_this_person_to_poi', 'shared_receipt_with_poi',
               ]
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print data_dict.keys()
print len(data_dict.keys())
print data_dict['METTS MARK']
print len(data_dict['METTS MARK'])


### Task 2: Remove outliers

data_dict.pop( 'TOTAL', 0 )
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

print len(data_dict)

### Task 3: Create new feature(s)



### Store to my_dataset for easy export below.
my_dataset = data_dict

# create a new feature: ratio of from_this_person_to_poi / from_messages
for key, value in data_dict.iteritems():

    if data_dict[key]["from_this_person_to_poi"] != "NaN" and data_dict[key]["from_messages"] != "NaN":
        my_dataset[key]["from_ratio"] = round(float(data_dict[key]["from_this_person_to_poi"])/data_dict[key]["from_messages"], 3)
    else:
        my_dataset[key]["from_ratio"] = "NaN"
print my_dataset['METTS MARK']["from_ratio"]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
my_feature_list = features_list

print my_feature_list

# k_best = SelectKBest(10)
# k_best.fit(features, labels)
# scores = k_best.scores_
# unsorted_pairs = zip(features_list[1:], scores)
# sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
# k_best_features = dict(sorted_pairs[:k])
# print k_best_features
num = 3

def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features.keys())
    return k_best_features

print get_k_best(my_dataset,all_features,num)

my_feature_list = ['poi'] + get_k_best(my_dataset,all_features,num).keys()


print my_feature_list

#print labels ,features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=3, p=2, weights='distance')

# from sklearn.cluster import KMeans
# k_clf = KMeans(n_clusters=2, tol=0.001)

# from sklearn.svm import SVC
# s_clf = SVC(kernel='rbf', C=1000)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
print features_list
from tester import test_classifier
test_classifier(clf, my_dataset, my_feature_list)

# test_classifier(k_clf, my_dataset, my_feature_list)

# test_classifier(s_clf, my_dataset, my_feature_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_feature_list)

# dump_classifier_and_data(k_clf, my_dataset, my_feature_list)