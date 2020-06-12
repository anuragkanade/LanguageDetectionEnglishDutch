import sys
import pickle
from make_features import Features
import adaboost as ada


def main():
    """
    runs the algorithm on test data using the model stored in file_name 'hypothesis' and prints out the results
    tree with root = None is a decision tree and tree without None is an adaboost dictionary

    first command line argument: file_name of file which has the serialized version of the model
    second command line argument: file_name of file which has the test_data

    :return: None
    """
    hypothesis = sys.argv[1]
    test_data = sys.argv[2]
    with open(hypothesis, "rb") as dt:
        dt_dict = pickle.load(dt)
    feature_value_mapping = []
    with open(test_data) as file:
        for line in file:
            features = Features()
            features.make_features(line)
            feature_value_mapping.append(features.features)

    if None in dt_dict.keys():
        results = decision_tree_predicter(feature_value_mapping, dt_dict)
    else:
        results = adaboost_predicter(dt_dict, feature_value_mapping)

    print(results)


def adaboost_predicter(dt_dict, feature_value_mapping):
    """
    Calls the adaboost predictor class

    :param dt_dict: model in the form of a dictionary
    :param feature_value_mapping: dictionary obtained from make_features
    :return: results of the language classification of the data
    """
    results = []
    for entry in feature_value_mapping:
        results.append(ada.decide(dt_dict, entry))
    return results


def decision_tree_predicter(dt_dict, feature_value_mapping):
    """
    returns the results of the decision tree predictions

    :param dt_dict: model in the form of a dictionary
    :param feature_value_mapping: dictionary obtained from make_features
    :return: results of the language classification of the data
    """
    results = []
    for entry in feature_value_mapping:
        key = dt_dict[None][0]
        while True:
            if key == "True:is_nl" or key == "False:is_nl" or key == "True:is_en" \
                    or key == "False:is_en":
                break
            else:
                if entry[key]:
                    key = dt_dict[key][0]
                else:
                    key = dt_dict[key][1]

        results.append(key.split(":")[1])
    return results


if __name__ == "__main__":
    main()
