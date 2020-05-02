import sys
import pickle
from make_features import Features
import adaboost as ada


def main():
    hypothesis = sys.argv[1]
    test_data = sys.argv[2]
    dt_dict = None
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
    results = []
    for entry in feature_value_mapping:
        results.append(ada.decide(dt_dict, entry))
    return results


def decision_tree_predicter(feature_value_mapping, dt_dict):
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
