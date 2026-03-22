import json
import pickle

import numpy as np


def load_parse_from_json(parse, setting_path):
    with open(setting_path, 'r') as f:
        setting = json.load(f)
    parse_dict = vars(parse)
    for k, v in setting.items():
        parse_dict[k] = v
    return parse


def get_aggregated_batch(data, batch_size, batch_no):
    return [data[d][batch_size * batch_no: batch_size * (batch_no + 1)] for d in range(len(data))]


def construct_list(data_dir, max_time_len):
    user, profile, itm_spar, itm_dens, label, pos, list_len = pickle.load(open(data_dir, 'rb'))
    print(len(user), len(itm_spar))
    cut_itm_dens, cut_itm_spar, cut_label, cut_pos = [], [], [], []
    for i, itm_spar_i, itm_dens_i, label_i, pos_i, list_len_i in zip(
            list(range(len(label))), itm_spar, itm_dens, label, pos, list_len):

        if len(itm_spar_i) >= max_time_len:
            cut_itm_spar.append(itm_spar_i[: max_time_len])
            cut_itm_dens.append(itm_dens_i[: max_time_len])
            cut_label.append(label_i[: max_time_len])
            cut_pos.append(pos_i[: max_time_len])
            list_len[i] = max_time_len
        else:
            cut_itm_spar.append(itm_spar_i + [np.zeros_like(np.array(itm_spar_i[0])).tolist()] * (max_time_len - len(itm_spar_i)))
            cut_itm_dens.append(itm_dens_i + [np.zeros_like(np.array(itm_dens_i[0])).tolist()] * (max_time_len - len(itm_dens_i)))
            cut_label.append(label_i + [0 for _ in range(max_time_len - list_len_i)])
            cut_pos.append(pos_i + [j for j in range(list_len_i, max_time_len)])

    return user, profile, cut_itm_spar, cut_itm_dens, cut_label, cut_pos, list_len


def evaluate_multi(labels, preds, scope_number, is_rank, _print=False):
    ndcg = [[] for _ in range(len(scope_number))]
    map = [[] for _ in range(len(scope_number))]
    clicks = [[] for _ in range(len(scope_number))]

    for label, pred in zip(labels, preds):
        if is_rank:
            final = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
        else:
            final = list(range(len(pred)))
        click = np.array(label)[final].tolist()
        gold = sorted(range(len(label)), key=lambda k: label[k], reverse=True)

        for i, scope in enumerate(scope_number):
            ideal_dcg, dcg, AP_value, AP_count = 0, 0, 0, 0
            cur_scope = min(scope, len(label))
            for _i, _g, _f in zip(range(1, cur_scope + 1), gold[:cur_scope], final[:cur_scope]):
                dcg += (pow(2, click[_i - 1]) - 1) / (np.log2(_i + 1))
                ideal_dcg += (pow(2, label[_g]) - 1) / (np.log2(_i + 1))

                if click[_i - 1] >= 1:
                    AP_count += 1
                    AP_value += AP_count / _i

            _ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.
            _map = float(AP_value) / AP_count if AP_count != 0 else 0.

            ndcg[i].append(_ndcg)
            map[i].append(_map)
            clicks[i].append(sum(click[:cur_scope]))

    return np.mean(np.array(map), axis=-1), np.mean(np.array(ndcg), axis=-1), np.mean(np.array(clicks), axis=-1), \
           [map, ndcg, clicks]
