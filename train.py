from make_features import Features
import decision_tree as dt
import adaboost as ada
import sys
import pickle


def main():
    examples = sys.argv[1]
    hypothesis_out = sys.argv[2]
    learning_type = sys.argv[3]
    training(examples, hypothesis_out, learning_type)


def training(file_name, hypothesis_out, learn_type):
    feature_value_mapping = []
    with open(file_name) as file:
        for line in file:
            features = Features()
            features.make_features(line)
            feature_value_mapping.append(features.features)

    if learn_type == "dt":
        dictionary = dt.make_decision_tree(feature_value_mapping)
    elif learn_type == "ada":
        dictionary = ada.make_stumps(feature_value_mapping)

    with open(hypothesis_out, "wb") as output_file:
        pickle.dump(dictionary, output_file)


if __name__ == "__main__":
    main()
