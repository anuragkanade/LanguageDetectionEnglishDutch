import decision_tree as dt
import math
import numpy as np
import copy


NUMBER_OF_STUMPS = 10
# Maximum is 10, equal to the number of features


def get_feature_list(temp_data_list):
    feature_list = []
    if temp_data_list:
        for i in range(len(temp_data_list[0].keys())):
            max_gain = 0
            max_gain_feat = None
            for key in temp_data_list[0]:
                if key == "res" or key == "sent" or key == "weight" or key == "sent":
                    continue
                feat_pos, feat_neg = dt.split_by_feat(temp_data_list, key)
                a_prob = dt.calculate_a_b_probability(temp_data_list)
                gain = dt.calculate_gain(a_prob, feat_pos, feat_neg)
                if gain > max_gain or max_gain_feat is None:
                    max_gain = gain
                    max_gain_feat = key
            if max_gain_feat is not None:
                feature_list.append(max_gain_feat)
                dt.remove_feat(temp_data_list, max_gain_feat)

    return feature_list


def make_stumps(data_list):
    new_data_list = copy.deepcopy(data_list)
    feat_list = get_feature_list(new_data_list)
    add_weights(data_list)
    feat_dict = {}
    count_stumps = 0
    if data_list:
        for key in feat_list:
            count_stumps += 1
            if count_stumps == NUMBER_OF_STUMPS:
                break
            if key == "res" or key == "sent" or key == "weight":
                continue
            new_data, amount_of_say = make_stump(data_list, key, feat_dict)
            change_weights(data_list, new_data, amount_of_say)
    print(feat_dict)
    return feat_dict


def change_weights(data_list, new_data, amount_of_say):
    total_weight = 0
    for i in range(len(new_data)):
        new_row = new_data[i]
        row = data_list[i]
        if new_row["incorrect"]:
            row["weight"] = row["weight"] * math.pow(math.e, amount_of_say)
            total_weight += row["weight"]
        else:
            row["weight"] = row["weight"] * math.pow(math.e, -amount_of_say)
            total_weight += row["weight"]
    factor = 1 / total_weight
    for row in data_list:
        row["weight"] = row["weight"] * factor


def make_stump(data_list, feat_name, feat_dict):
    new_data = get_new_stump_data(data_list, feat_name)
    feat_pos, feat_neg = dt.split_by_feat(new_data, feat_name)
    is_nl_prob_feat_pos = dt.calculate_a_b_probability(feat_pos)
    is_nl_prob_feat_neg = dt.calculate_a_b_probability(feat_neg)
    amount_of_say = find_amount_of_say(new_data, feat_name, is_nl_prob_feat_pos, is_nl_prob_feat_neg)
    add_child(feat_dict, feat_name, "amount_of_say", amount_of_say)
    if is_nl_prob_feat_pos >= 0.5:
        add_child(feat_dict, feat_name, "True", "is_nl")
    else:
        add_child(feat_dict, feat_name, "True", "is_en")

    if is_nl_prob_feat_neg >= 0.5:
        add_child(feat_dict, feat_name, "False", "is_nl")
    else:
        add_child(feat_dict, feat_name, "False", "is_en")

    return new_data, amount_of_say


def find_amount_of_say(new_data, feat_name, is_nl_prob_pos, is_nl_prob_neg):
    error = find_error(new_data, feat_name, is_nl_prob_pos, is_nl_prob_neg)
    factor = (1 - error) / (error + np.finfo(float).eps)
    return 0.5 * math.log(factor, math.e)


def find_error(data, feat_name, is_nl_prob_pos, is_nl_prob_neg):
    error = 0.0
    pos_conclusion = False
    neg_conclusion = False
    if is_nl_prob_pos >= 0.5:
        pos_conclusion = True
    if is_nl_prob_neg >= 0.5:
        neg_conclusion = True
    for row in data:
        if row[feat_name] and row["res"] != pos_conclusion:
            error += row["weight"]
            row["incorrect"] = True
        elif not row[feat_name] and row["res"] != neg_conclusion:
            error += row["weight"]
            row["incorrect"] = True
        else:
            row["incorrect"] = False

    return error


def add_weights(data_list):
    count_sample = len(data_list)
    for data in data_list:
        data["weight"] = 1 / count_sample


def get_new_stump_data(data_list, feat_name):
    new_data = []
    for data in data_list:
        new_data.append({feat_name: data[feat_name],
                         "res": data["res"], "weight": data["weight"]})
    return new_data


def add_child(feat_dict, parent_feat, child_feat, final_outcome):
    append_string = child_feat
    if child_feat == "True" or child_feat == "False":
        append_string += ":" + final_outcome
    if child_feat == "amount_of_say":
        append_string += ":" + str(final_outcome)
    if parent_feat not in feat_dict.keys():
        feat_dict[parent_feat] = [append_string]
    else:
        feat_dict[parent_feat].append(append_string)
