from make_features import Features
import decision_tree as dt
import adaboost as ada
import sys
import pickle


def main():
    """
    The main function called for training the model
    first command line argument: file_name which has the training data
    second command line argument: file_name which will store the serialized model
    third command line argument: if the model to be trained is decision tree or adaboost

    :return: None
    """
    examples = sys.argv[1]
    hypothesis_out = sys.argv[2]
    learning_type = sys.argv[3]
    training(examples, hypothesis_out, learning_type)


def training(input_file_name, hypothesis_out, learn_type):
    """
    calls make_features.py to make a dictionary of features and passes it on
    transfers control to the respective learning model asked
    after model is returned the dictionary is written is serialized using pickle and stored in the hypothesis_out file

    :param input_file_name: file_name of input data
    :param hypothesis_out: file_name in which model is to be stored
    :param learn_type: can be 'dt' for decision tree or 'ada' for adaboost

    :return: None
    """
    feature_value_mapping = []
    dictionary = {}
    with open(input_file_name) as file:
        for line in file:
            features = Features()
            features.make_features(line)
            feature_value_mapping.append(features.features)

    if learn_type == "dt":
        hypothesis_out = "dt" + hypothesis_out
        dictionary = dt.make_decision_tree(feature_value_mapping)
    elif learn_type == "ada":
        hypothesis_out = "ada" + hypothesis_out
        dictionary = ada.make_stumps(feature_value_mapping)

    print(dictionary)
    with open(hypothesis_out, "wb") as output_file:
        pickle.dump(dictionary, output_file)


if __name__ == "__main__":
    main()
