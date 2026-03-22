import os
import time
import random
import json
import argparse
import datetime
import pickle as pkl

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import construct_list, get_aggregated_batch, evaluate_multi, load_parse_from_json

from prm_model import PRMModel


def eval_model(model, data, batch_size, is_rank, metric_scope, device):
    model.eval()
    preds = []
    losses = []

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    t = time.time()
    with torch.no_grad():
        for batch_no in range(batch_num):
            batch_data = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
            pred, loss = forward_batch(model, batch_data, device)
            preds.extend(pred)
            losses.append(loss)

    loss = sum(losses) / len(losses)
    labels = data[4]
    res = evaluate_multi(labels, preds, metric_scope, is_rank)

    print("EVAL TIME: %.4fs" % (time.time() - t))
    return loss, res


def forward_batch(model, batch_data, device):
    """Run forward pass on a batch, return predictions and loss."""
    # batch_data: [user, profile, itm_spar, itm_dens, label, pos, list_len]
    itm_spar = torch.LongTensor(np.array(batch_data[2])).to(device)
    itm_dens = torch.FloatTensor(np.array(batch_data[3])).to(device)
    labels = torch.FloatTensor(np.array(batch_data[4])).to(device)
    seq_length = torch.LongTensor(np.array(batch_data[6])).to(device)

    y_pred = model(itm_spar, itm_dens, seq_length)

    # Log loss: -(y*log(p) + (1-y)*log(1-p)), averaged
    eps = 1e-7
    y_pred_clamped = torch.clamp(y_pred, eps, 1 - eps)
    loss = -torch.mean(labels * torch.log(y_pred_clamped) + (1 - labels) * torch.log(1 - y_pred_clamped))

    pred_list = y_pred.cpu().numpy().reshape(-1, model.max_time_len).tolist()
    return pred_list, loss.item()


def train_model(train_file, test_file, feature_size, max_time_len,
                itm_spar_fnum, itm_dens_fnum, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dropout = 1.0 - params.keep_prob

    model = PRMModel(
        feature_size=feature_size,
        eb_dim=params.eb_dim,
        max_time_len=max_time_len,
        itm_spar_num=itm_spar_fnum,
        itm_dens_num=itm_dens_fnum,
        d_model=64,
        d_inner_hid=128,
        n_head=1,
        dropout=dropout,
    ).to(device)

    # L2 regularization via weight_decay on non-bias, non-embedding params
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if 'bias' in name or 'emb_mtx' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.Adam([
        {'params': decay_params, 'weight_decay': params.l2_reg},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=params.lr)

    # Naming and directories
    model_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
        params.timestamp, params.initial_ranker, params.model_type,
        params.batch_size, params.lr, params.l2_reg,
        params.eb_dim, params.keep_prob)

    save_model_dir = '{}/save_model_{}/{}/{}/'.format(
        params.save_dir, params.data_set_name, max_time_len, model_name)
    log_dir = '{}/logs_{}/{}/'.format(
        params.save_dir, params.data_set_name, max_time_len)
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    save_path = os.path.join(save_model_dir, 'best_model.pt')
    log_save_path = os.path.join(log_dir, model_name + '.metrics')

    training_monitor = {
        'train_loss': [],
        'vali_loss': [],
        'map_l': [],
        'ndcg_l': [],
        'clicks_l': [],
    }

    # Initial evaluation (before training, no ranking = initial ranker order)
    step = 0
    vali_loss, res = eval_model(model, test_file, params.batch_size, False, params.metric_scope, device)

    training_monitor['train_loss'].append(None)
    training_monitor['vali_loss'].append(None)
    training_monitor['map_l'].append(res[0][0])
    training_monitor['ndcg_l'].append(res[1][0])
    training_monitor['clicks_l'].append(res[2][0])

    print("STEP %d  INITIAL RANKER | LOSS VALI: NULL" % step)
    for i, s in enumerate(params.metric_scope):
        print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f" % (s, res[0][i], res[1][i], res[2][i]))

    # Training
    data_size = len(train_file[0])
    batch_num = data_size // params.batch_size
    eval_iter_num = (data_size // 5) // params.batch_size
    print('train', data_size, batch_num)

    train_losses_step = []
    early_stop = False

    for epoch in range(params.epoch_num):
        model.train()
        for batch_no in range(batch_num):
            batch_data = get_aggregated_batch(train_file, batch_size=params.batch_size, batch_no=batch_no)

            itm_spar = torch.LongTensor(np.array(batch_data[2])).to(device)
            itm_dens = torch.FloatTensor(np.array(batch_data[3])).to(device)
            labels = torch.FloatTensor(np.array(batch_data[4])).to(device)
            seq_length = torch.LongTensor(np.array(batch_data[6])).to(device)

            optimizer.zero_grad()
            y_pred = model(itm_spar, itm_dens, seq_length)

            # Log loss
            eps = 1e-7
            y_pred_clamped = torch.clamp(y_pred, eps, 1 - eps)
            loss = -torch.mean(labels * torch.log(y_pred_clamped) + (1 - labels) * torch.log(1 - y_pred_clamped))

            loss.backward()

            # Gradient clipping
            if params.max_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), params.max_norm)

            optimizer.step()
            step += 1
            train_losses_step.append(loss.item())

            if step % eval_iter_num == 0:
                train_loss = sum(train_losses_step) / len(train_losses_step)
                train_losses_step = []

                vali_loss, res = eval_model(model, test_file, params.batch_size, True, params.metric_scope, device)

                training_monitor['train_loss'].append(train_loss)
                training_monitor['vali_loss'].append(vali_loss)
                training_monitor['map_l'].append(res[0][0])
                training_monitor['ndcg_l'].append(res[1][0])
                training_monitor['clicks_l'].append(res[2][0])

                print("EPOCH %d STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f" % (epoch, step, train_loss, vali_loss))
                for i, s in enumerate(params.metric_scope):
                    print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f" % (s, res[0][i], res[1][i], res[2][i]))

                # Save best model by MAP
                if training_monitor['map_l'][-1] > max(training_monitor['map_l'][:-1]):
                    torch.save(model.state_dict(), save_path)
                    pkl.dump(res[-1], open(log_save_path, 'wb'))
                    print('model saved')

                # Early stopping
                if len(training_monitor['map_l']) > 2 and epoch > 0:
                    if ((training_monitor['map_l'][-2] - training_monitor['map_l'][-1]) <= 0.01 and
                            (training_monitor['map_l'][-3] - training_monitor['map_l'][-2]) <= 0.01):
                        early_stop = True

                model.train()

        # Save training monitor
        with open(os.path.join(log_dir, model_name + '.monitor.pkl'), 'wb') as f:
            pkl.dump(training_monitor, f)

        if early_stop:
            print('Early stopping at epoch %d' % epoch)
            break


def reranker_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_time_len', default=10, type=int, help='max time length')
    parser.add_argument('--save_dir', type=str, default='./', help='dir that saves logs and model')
    parser.add_argument('--data_dir', type=str, default='./data/toy/', help='data dir')
    parser.add_argument('--model_type', default='PRM', type=str)
    parser.add_argument('--data_set_name', default='ad', type=str)
    parser.add_argument('--initial_ranker', default='lambdaMART',
                        choices=['DNN', 'lambdaMART'], type=str)
    parser.add_argument('--epoch_num', default=30, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--l2_reg', default=1e-4, type=float)
    parser.add_argument('--keep_prob', default=0.8, type=float)
    parser.add_argument('--eb_dim', default=16, type=int)
    parser.add_argument('--metric_scope', default=[1, 3, 5, 10], type=list)
    parser.add_argument('--max_norm', default=0, type=float)
    parser.add_argument('--timestamp', type=str,
                        default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument('--setting_path', type=str,
                        default='./config/prm_setting.json')
    FLAGS, _ = parser.parse_known_args()
    return FLAGS


if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    parse = reranker_parse_args()
    if parse.setting_path:
        parse = load_parse_from_json(parse, parse.setting_path)

    data_set_name = parse.data_set_name
    processed_dir = parse.data_dir
    stat_dir = os.path.join(processed_dir, 'data.stat')
    max_time_len = parse.max_time_len
    initial_ranker = parse.initial_ranker

    print(parse)

    with open(stat_dir, 'r') as f:
        stat = json.load(f)

    num_ft = stat['ft_num']
    itm_spar_fnum = stat['itm_spar_fnum']
    itm_dens_fnum = stat['itm_dens_fnum']

    print('num of item', stat['item_num'],
          'num of list', stat['train_num'] + stat['val_num'] + stat['test_num'],
          'spar num', itm_spar_fnum,
          'dens num', itm_dens_fnum)

    # Construct training data
    train_dir = os.path.join(processed_dir, initial_ranker + '.data.train')
    if os.path.isfile(train_dir):
        train_lists = pkl.load(open(train_dir, 'rb'))
    else:
        train_lists = construct_list(
            os.path.join(processed_dir, initial_ranker + '.rankings.train'), max_time_len)
        pkl.dump(train_lists, open(train_dir, 'wb'))

    # Construct test data
    test_dir = os.path.join(processed_dir, initial_ranker + '.data.test')
    if os.path.isfile(test_dir):
        test_lists = pkl.load(open(test_dir, 'rb'))
    else:
        test_lists = construct_list(
            os.path.join(processed_dir, initial_ranker + '.rankings.test'), max_time_len)
        pkl.dump(test_lists, open(test_dir, 'wb'))

    train_model(train_lists, test_lists, num_ft, max_time_len,
                itm_spar_fnum, itm_dens_fnum, parse)
