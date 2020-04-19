from make_features import Features


def main():
    training("train.dat")


def training(file_name):
    with open(file_name) as file:
        for line in file:
            features = Features()

            features.make_features(line)
            print(features)


if __name__ == "__main__":
    main()
