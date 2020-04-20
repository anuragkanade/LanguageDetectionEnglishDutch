import math
import numpy as np


def entropy(probability):
    q = 1 - probability
    return -(probability * math.log(probability + np.finfo(float).eps, 2) + q * math.log(q + np.finfo(float).eps, 2))


def split_by_feat(data_list, feat_name):
    feat_pos = []
    feat_neg = []
    for i in data_list:
        if i[feat_name]:
            feat_pos.append(i)
        else:
            feat_neg.append(i)
    return feat_pos, feat_neg


def calculate_a_b_counts(data_list):
    a_count = 0
    b_count = 0
    for i in data_list:
        if i["res"]:
            a_count += 1
        else:
            b_count += 1
    return a_count, b_count


def calculate_probability(a_count, b_count):
    return a_count / (a_count + b_count + np.finfo(float).eps)


def calculate_a_b_probability(data_list):
    a_count, b_count = calculate_a_b_counts(data_list)
    return a_count / (a_count + b_count)


def file_reader(filename):
    lines = []
    with open(filename) as file:
        j = 0
        for line in file:
            line = line.strip()
            line_list = line.split(" ")
            lines.append(line_list)
            j += 1
    return lines


def remainder(feat_pos, feat_neg):
    a_count_pos, b_count_pos = calculate_a_b_counts(feat_pos)
    a_count_neg, b_count_neg = calculate_a_b_counts(feat_neg)
    count_ratio_pos = (a_count_pos + b_count_pos) / (len(feat_pos) + len(feat_neg))
    count_ratio_neg = (a_count_neg + b_count_neg) / (len(feat_pos) + len(feat_neg))
    sum_prob = count_ratio_pos * entropy(calculate_probability(a_count_pos, b_count_pos)) + \
               count_ratio_neg * entropy(calculate_probability(a_count_neg, b_count_neg))
    return sum_prob


def information_gain(data_list):
    max_gain = 0
    first_feat = None
    gains = []
    if not data_list:
        return
    for key in data_list[0].keys():
        feat_pos, feat_neg = split_by_feat(data_list, key)
        a_prob = calculate_a_b_probability(data_list)
        entrop = entropy(a_prob)
        rem = remainder(feat_pos, feat_neg)
        gain = entrop - rem
        gains.append(gain)
        if gain > max_gain:
            max_gain = gain
            first_feat = key
    return first_feat


def map_feat_index(data_list):
    feat_dict = {}
    for data in data_list:
        j = 0
        for key in data.keys():
            feat_dict[j] = key
            j += 1
    return feat_dict


def remove_feat(data_list, feat_name):
    for data in data_list:
        data.remove(feat_name)

    return data_list


def make_decision_tree(data_list):
    feat_list = []
    # print(data_list)
    feat_dict = map_feat_index(data_list)
    # print(feat_dict)
    data_process(data_list, feat_list, feat_dict)
    print(feat_list)

    # print(calculate_a_b_counts(feat_pos), "True  - probability")
    # print(calculate_a_b_counts(feat_neg), "False probability")
    # feat_index = information_gain(feat_pos)
    # feat_list.append(feat_dict[feat_index])
    # feat_pos_2, feat_neg_2 = split_by_feat(feat_pos, feat_index)
    # a_count, b_count = calculate_a_b_counts(feat_pos_2)
    # print(a_count, b_count, "True - True", sep=" ")
    # a_count, b_count = calculate_a_b_counts(feat_neg_2)
    # print(a_count, b_count, "True - False", sep=" ")
    # print(calculate_a_b_probability(feat_pos_2), "True - True", sep=" ")
    # print(calculate_a_b_probability(feat_neg_2), "True - False", sep=" ")

    # feat_index = information_gain(feat_neg)
    # feat_list.append(feat_dict[feat_index])
    # feat_pos_3, feat_neg_3 = split_by_feat(feat_neg, feat_index)
    # a_count, b_count = calculate_a_b_counts(feat_pos_3)
    # print(a_count, b_count, "False - True", sep=" ")
    # a_count, b_count = calculate_a_b_counts(feat_neg_3)
    # print(a_count, b_count, "False - False", sep=" ")
    # print(calculate_a_b_probability(feat_pos_3), "False - True", sep=" ")
    # print(calculate_a_b_probability(feat_neg_3), "False - False", sep=" ")
    return feat_list


def get_new_dictionary(feat_dict, feat_index):
    new_dict = {}
    for key, value in feat_dict.items():
        if key < feat_index:
            new_dict[key] = feat_dict[key]
        elif key > feat_index:
            new_dict[key - 1] = feat_dict[key]
    return new_dict


def data_process(data, feat_list, feat_dict):
    feat_name = information_gain(data)
    if feat_name is None:
        return
    feat_list.append(feat_name)
    feat_pos, feat_neg = split_by_feat(data, feat_name)
    prob_pos = calculate_a_b_probability(feat_pos)
    prob_neg = calculate_a_b_probability(feat_neg)
    a_count, b_count = calculate_a_b_counts(feat_pos)
    print(a_count, b_count, sep=" ")
    a_count, b_count = calculate_a_b_counts(feat_neg)
    print(a_count, b_count, sep=" ")
    if prob_pos != 0 and prob_pos != 1:
        feat_pos = remove_feat(feat_pos, feat_name)
        data_process(feat_pos, feat_list, feat_dict)

    if prob_neg != 0 and prob_neg != 1:
        feat_neg = remove_feat(feat_neg, feat_name)
        data_process(feat_neg, feat_list, feat_dict)


def main():
    data_list = file_reader("dtree-data.txt")
    print(make_decision_tree(data_list))


if __name__ == "__main__":
    main()
