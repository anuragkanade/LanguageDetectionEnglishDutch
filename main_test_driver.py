from make_features import Features
import decision_tree as dt


def main():
    training("train.dat")


def training(file_name):
    feature_value_mapping = []
    with open(file_name) as file:
        for line in file:
            features = Features()
            features.make_features(line)
            feature_value_mapping.append(features.features)

    dt.make_decision_tree(feature_value_mapping)



if __name__ == "__main__":
    main()
