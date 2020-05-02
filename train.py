from make_features import Features
import decision_tree as dt
import sys
import pickle


def main():
    examples = sys.argv[1]
    hypothesis_out = sys.argv[2]
    learning_type = sys.argv[3]
    if learning_type == "dt":
        training(examples, hypothesis_out)


def training(file_name, hypothesis_out):
    feature_value_mapping = []
    with open(file_name) as file:
        for line in file:
            features = Features()
            features.make_features(line)
            feature_value_mapping.append(features.features)

    dictionary = dt.make_decision_tree(feature_value_mapping)
    print(dictionary)
    with open(hypothesis_out, "wb") as output_file:
        pickle.dump(dictionary, output_file)


if __name__ == "__main__":
    main()
