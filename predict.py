import sys
import pickle
from make_features import Features


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

    results = []
    for entry in feature_value_mapping:
        key = dt_dict[None][0]
        while True:
            if key == "True:is_nl" or key == "False:is_nl" or key == "True:is_en" \
                    or key == "False:is_en":
                break
                key = dt_dict[key][0]
            else:
                key = dt_dict[key][1]

        results.append(key.split(":")[1])
    print(results)


if __name__ == "__main__":
    main()
