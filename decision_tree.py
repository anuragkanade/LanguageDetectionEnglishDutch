import math
import numpy as np


# maximum depth of the decision tree being created
MAX_DEPTH = math.inf


def entropy(probability):
    """
    calculates the entropy given a probability of an event occurring

    :param probability: probability of an event occurring
    :return: corresponding entropy
    """
    q = 1 - probability
    return -(probability * math.log(probability + np.finfo(float).eps, 2) + q * math.log(q + np.finfo(float).eps, 2))


def split_by_feat(data_list, feat_name):
    """
    splits the data according to value of the feature specified in feat_name

    :param data_list: complete data with all features
    :param feat_name: feature to be conidered
    :return:    feat_pos: data where value of specified feature is True
                feat_neg: data where value of specified feature is False
    """
    feat_pos = []
    feat_neg = []
    for i in data_list:
        if i[feat_name]:
            feat_pos.append(i)
        else:
            feat_neg.append(i)
    return feat_pos, feat_neg


def calculate_a_b_counts(data_list):
    """
    returns count of is_nl and count of is_en from data list

    :param data_list: the entire data
    :return:  count of is_en and is_nl
    """
    a_count = 0
    b_count = 0
    for i in data_list:
        if i["res"]:
            a_count += 1
        else:
            b_count += 1
    return a_count, b_count


def calculate_probability(a_count, b_count):
    """
    returns probability of a occurring, in the case of the project that is is_nl

    :param a_count: count of a
    :param b_count: count of b

    :return: probability of a occurring
    """
    return a_count / (a_count + b_count + np.finfo(float).eps)


def calculate_a_b_probability(data_list):
    """
    returns probability of a occurring, in the case of the project that is is_nl
    scans the data and returns the probability

    :param data_list: entire data

    :return: probability of a occurring
    """
    a_count, b_count = calculate_a_b_counts(data_list)
    return a_count / (a_count + b_count + np.finfo(float).eps)


def file_reader(filename):
    """
    reads file specified in filename

    :param filename: name of files
    :return: returns a list of lines in the file
    """
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
    """
    calculates remaining entropy after splitting by feature

    :param feat_pos: data where feature value is True
    :param feat_neg: data where feature value is False

    :return: remaining entropy
    """
    a_count_pos, b_count_pos = calculate_a_b_counts(feat_pos)
    a_count_neg, b_count_neg = calculate_a_b_counts(feat_neg)
    count_ratio_pos = (a_count_pos + b_count_pos) / (len(feat_pos) + len(feat_neg))
    count_ratio_neg = (a_count_neg + b_count_neg) / (len(feat_pos) + len(feat_neg))
    sum_prob = count_ratio_pos * entropy(calculate_probability(a_count_pos, b_count_pos)) + \
               count_ratio_neg * entropy(calculate_probability(a_count_neg, b_count_neg))
    return sum_prob


def find_feature_with_highest_information_gain(data_list):
    """
    calculates information gain for all possible feature splits and returns the name of the feature
    with highest gain

    :param data_list: all data

    :return: name of feature
    """
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
        gain = calculate_gain(a_prob, feat_pos, feat_neg)
        gains.append(gain)
        if gain > max_gain:
            max_gain = gain
            first_feat = key
    return first_feat


def calculate_gain(a_prob, feat_pos, feat_neg):
    """
    Calculates information gain if data with True feature value and data with False feature value are given

    :param a_prob:   probability o is_en
    :param feat_pos: data with feat_value = True
    :param feat_neg: data with feat_value = False

    :return: calculated gain for the data partitions given
    """
    entrop = entropy(a_prob)
    rem = remainder(feat_pos, feat_neg)
    gain = entrop - rem
    return gain


def remove_feat(data_list, feat_name):
    """
    removes feature from given data if present

    :param data_list: input data
    :param feat_name: name of feature to be removed

    :return: the edited data
    """
    for data in data_list:
        data.pop(feat_name)

    return data_list


def make_decision_tree(data_list):
    """
    setup function which creates the empty dictionary and calls the recursive method
    data_process

    :param data_list: input data

    :return: filled dictionary with the parent as the key and a list as the value
    children are elements of the list
    """
    feat_dict = {}
    data_process(data_list, feat_dict, None, MAX_DEPTH)

    return feat_dict


def data_process(data, feat_dict, parent_feat, depth):
    """
    recursive function where the tree dictionary is populated

    :param data: input data
    :param feat_dict: current state of dictionary
    :param parent_feat: feature which you are finding children of
    :param depth: depth levels left to explore, till we reach MAX_DEPTH

    :return: None
    """

    feat_name = find_feature_with_highest_information_gain(data)
    if feat_name is None:
        return
    if depth == 0:
        return True

    add_child(feat_dict, parent_feat, feat_name, len(data))
    feat_pos, feat_neg = split_by_feat(data, feat_name)
    prob_pos = calculate_a_b_probability(feat_pos)
    prob_neg = calculate_a_b_probability(feat_neg)

    # if we know the probability is 0 or 1, we can stop

    if prob_pos != 0 and prob_pos != 1:
        feat_pos = remove_feat(feat_pos, feat_name)
        is_fin = data_process(feat_pos, feat_dict, feat_name, depth - 1)
        if is_fin:
            if prob_pos >= 0.5:
                add_child(feat_dict, feat_name, "True", "is_nl")
            else:
                add_child(feat_dict, feat_name, "True", "is_en")

    elif prob_pos == 1:
        add_child(feat_dict, feat_name, "True", "is_nl")
    else:
        add_child(feat_dict, feat_name, "True", "is_en")

    # if the probability is 0 or 1, we can stop

    if prob_neg != 0 and prob_neg != 1:
        feat_neg = remove_feat(feat_neg, feat_name)
        is_fin = data_process(feat_neg, feat_dict, feat_name, depth - 1)
        if is_fin:
            if prob_neg >= 0.5:
                add_child(feat_dict, feat_name, "False", "is_nl")
            else:
                add_child(feat_dict, feat_name, "False", "is_en")

    elif prob_neg == 1:
        add_child(feat_dict, feat_name, "False", "is_nl")
    else:
        add_child(feat_dict, feat_name, "False", "is_en")


def add_child(feat_dict, parent_feat, child_feat, final_outcome):
    """
    add child to the dictionary key specified in the parent_feat variable

    :param feat_dict: current tree dictionary
    :param parent_feat: node to which children are being added
    :param child_feat: node to which children are being added
    :param final_outcome: result if child is chosen, can be is_nl or is_en

    :return: None
    """

    append_string = child_feat
    if child_feat == "True" or child_feat == "False":
        append_string += ":" + final_outcome
    if parent_feat not in feat_dict.keys():
        feat_dict[parent_feat] = [append_string]
    else:
        feat_dict[parent_feat].append(append_string)

