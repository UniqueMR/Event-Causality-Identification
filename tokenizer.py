# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:28:57 2022

@author: Jason
"""
import torch
from transformers import RobertaTokenizer
from parameter import parse_args
args = parse_args()  # load parameters
tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

# delete tab
def delete_tokens(arg_idx, arg_mask, index, idx):
    arg_idx = torch.LongTensor(arg_idx)
    temp = torch.nonzero(arg_idx == idx, as_tuple=False)
    indices = temp[index][1]
    arg_i = torch.cat((arg_idx[0][0:indices], arg_idx[0][indices + 1:]))
    arg_i = torch.unsqueeze(arg_i, dim=0)
    arg_m = torch.cat((arg_mask[0][0:indices], arg_mask[0][indices + 1:]))
    arg_m = torch.unsqueeze(arg_m, dim=0)
    return arg_i, arg_m, indices


# tokenize sentence and get event idx
def get_batch(arg, indices):
    batch_idx = []
    batch_mask = []
    batch_event = []
    label_b = []
    clabel_b = []
    for idx in indices:
        label_1 = 1
        clabel_1 = 1
        front = 1
        e1_id = arg[idx][14]
        e2_id = arg[idx][15]
        sentence1_id = arg[idx][11]
        sentence2_id = arg[idx][13]
        s_1 = arg[idx][10]
        s_2 = arg[idx][12]
        s_1 = s_1.split()[0:int(args.plm_len/2)]
        s_2 = s_2.split()[0:int(args.plm_len/2)]
        e1_id = e1_id.split("_")
        e2_id = e2_id.split("_")
        if sentence1_id == sentence2_id:
            clabel_1 = 0
            if int(e1_id[1]) > int(e2_id[1]):
                s_1.insert(int(e1_id[1]), '<s>')
                s_1.insert(int(e1_id[1]) + len(e1_id), '<s>')
                s_1.insert(int(e2_id[1]), '<s>')
                s_1.insert(int(e2_id[1]) + len(e2_id), '<s>')
                front = 2
            else:
                s_1.insert(int(e2_id[1]) , '<s>')
                s_1.insert(int(e2_id[1]) + len(e2_id), '<s>')
                s_1.insert(int(e1_id[1]), '<s>')
                s_1.insert(int(e1_id[1]) + len(e1_id), '<s>')
                front = 1
            s_1 = " ".join(s_1)
            s_1 = s_1.replace(' <s>', '<s>')
            if int(e1_id[1]) == 0 or int(e2_id[1]) == 0:
                s_1 = s_1.replace('<s> ', '<s>', 1)
            encode_dict = tokenizer.encode_plus(
                s_1,
                add_special_tokens=True,
                padding='max_length',
                max_length=args.plm_len,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        else:
            s_1.insert(int(e1_id[1]), '<s>')
            s_1.insert(int(e1_id[1]) + len(e1_id), '<s>')
            s_2.insert(int(e2_id[1]), '<s>')
            s_2.insert(int(e2_id[1]) + len(e2_id), '<s>')
            s_1 = " ".join(s_1)
            s_2 = " ".join(s_2)
            s_1 = s_1.replace(' <s>', '<s>')
            s_2 = s_2.replace(' <s>', '<s>')
            if int(e1_id[1]) == 0:
                s_1 = s_1.replace('<s> ', '<s>', 1)
            if int(e2_id[1]) == 0:
                s_2 = s_2.replace('<s> ', '<s>', 1)
            if sentence1_id < sentence2_id:
                front = 1
                encode_dict = tokenizer.encode_plus(
                    s_1,
                    text_pair=s_2,
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=args.plm_len,
                    truncation=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
            else:
                front = 2
                encode_dict = tokenizer.encode_plus(
                    s_2,
                    text_pair=s_1,
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=args.plm_len,
                    truncation=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
        arg_1_idx = encode_dict['input_ids']
        arg_1_mask = encode_dict['attention_mask']
        if clabel_1 == 0:
            if front == 1:
                arg_1_idx, arg_1_mask, v1 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
                arg_1_idx, arg_1_mask, v2 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
                arg_1_idx, arg_1_mask, v3 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
                arg_1_idx, arg_1_mask, v4 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
            else:
                arg_1_idx, arg_1_mask, v3 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
                arg_1_idx, arg_1_mask, v4 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
                arg_1_idx, arg_1_mask, v1 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
                arg_1_idx, arg_1_mask, v2 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
        else:
            if front == 1:
                arg_1_idx, arg_1_mask, v1 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
                arg_1_idx, arg_1_mask, v2 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
                arg_1_idx, arg_1_mask, v3 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
                arg_1_idx, arg_1_mask, v4 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
            else:
                arg_1_idx, arg_1_mask, v3 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
                arg_1_idx, arg_1_mask, v4 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
                arg_1_idx, arg_1_mask, v1 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
                arg_1_idx, arg_1_mask, v2 = delete_tokens(arg_1_idx, arg_1_mask, 1, 0)
        arg_e_idx1 = torch.tensor([[v1, v2, v3, v4]])
        if arg[idx][9] == 'NONE':
            label_1 = 0
        label_b.append(label_1)
        clabel_b.append(clabel_1)
        if len(batch_idx) == 0:
            batch_idx = arg_1_idx
            batch_mask = arg_1_mask
            batch_event = arg_e_idx1
        else:
            batch_idx = torch.cat((batch_idx, arg_1_idx), dim=0)
            batch_mask = torch.cat((batch_mask, arg_1_mask), dim=0)
            batch_event = torch.cat((batch_event, arg_e_idx1), dim=0)
    return batch_idx, batch_mask, batch_event, label_b, clabel_b