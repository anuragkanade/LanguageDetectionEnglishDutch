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
        if key == "res" or key == "sent":
            continue
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
        data.pop(feat_name)

    return data_list


def make_decision_tree(data_list):
    feat_dict = {}
    # print(data_list)
    # print(feat_dict)
    data_process(data_list, feat_dict, None)
    print(feat_dict)

    return feat_dict


def get_new_dictionary(feat_dict, feat_index):
    new_dict = {}
    for key, value in feat_dict.items():
        if key < feat_index:
            new_dict[key] = feat_dict[key]
        elif key > feat_index:
            new_dict[key - 1] = feat_dict[key]
    return new_dict


def data_process(data, feat_dict, parent_feat, branch_string="None"):
    # Anfield case breaks stuff. No current attribute improves stuff in that case, as even the dutch wiki page has an
    # english article
    feat_name = information_gain(data)
    if feat_name is None:
        return
    add_child(feat_dict, parent_feat, feat_name, branch_string, len(data))
    feat_pos, feat_neg = split_by_feat(data, feat_name)
    prob_pos = calculate_a_b_probability(feat_pos)
    prob_neg = calculate_a_b_probability(feat_neg)
    a_count, b_count = calculate_a_b_counts(feat_pos)
    print(feat_name, "True", a_count, b_count, sep=" ")
    a_count, b_count = calculate_a_b_counts(feat_neg)
    print(feat_name, "False", a_count, b_count, sep=" ")
    if prob_pos != 0 and prob_pos != 1:
        feat_pos = remove_feat(feat_pos, feat_name)
        data_process(feat_pos, feat_dict, feat_name, "True")
    elif prob_pos == 1:
        add_child(feat_dict, feat_name, "True", "Leaf", "is_nl")
        # True means is_nl
    else:
        add_child(feat_dict, feat_name, "True", "Leaf", "is_en")
        # False means is_en

    if prob_neg != 0 and prob_neg != 1:
        feat_neg = remove_feat(feat_neg, feat_name)
        data_process(feat_neg, feat_dict, feat_name, "False")
    elif prob_neg == 1:
        add_child(feat_dict, feat_name, "False", "Leaf", "is_nl")
    else:
        add_child(feat_dict, feat_name, "False", "Leaf", "is_en")


def add_child(feat_list, parent_feat, child_feat, branch_string, length_data):
    append_string = branch_string + "_" + child_feat + "_" + str(length_data)
    if parent_feat not in feat_list.keys():
        feat_list[parent_feat] = [append_string]
    else:
        feat_list[parent_feat].append(append_string)


def main():
    data_list = file_reader("dtree-data.txt")
    print(make_decision_tree(data_list))


if __name__ == "__main__":
    main()
