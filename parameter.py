# -*- coding: utf-8 -*-

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ECI')

    parser.add_argument('--model_name',       default='roberta-base',   type=str,     help='Log model name')
    parser.add_argument('--embedding_size',   default=768,              type=int,     help='PLM embedding size')
    parser.add_argument('--plm_len',          default=200,              type=int,     help='PLM max padding length')
    parser.add_argument('--mlp_size',         default=2,                type=int,     help='mlp layer_size')
    parser.add_argument('--seed',             default=2022,             type=int,     help='seed for reproducibility')
    parser.add_argument('--batch_size',       default=16,               type=int,     help='batchsize for optimizer updates')
    parser.add_argument('--num_epoch',        default=10,               type=int,     help='number of total epochs to run')
    parser.add_argument('--lr',               default=5e-6,             type=float,   help='initial learning rate')
    parser.add_argument('--log',              default='./out/',         type=str,     help='Log result file name')
    
    args = parser.parse_args()
    return args
