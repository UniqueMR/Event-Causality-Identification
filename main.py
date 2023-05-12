# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import torch
import torch.nn as nn
import logging
import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from load_data import load_data
from model import Base,modified_with_lstm, modified_with_attention_mask
from tokenizer import get_batch
from parameter import parse_args
args = parse_args()  # load parameters
writer = SummaryWriter()

# ---------- GPU set ---------- 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# ---------- create results file ---------- 
if not os.path.exists(args.log):
    os.mkdir(args.log) 
t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + t + '.txt'

# ---------- Run logging Set ---------- 
for name in logging.root.manager.loggerDict:
    if 'transformers' in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)
logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=args.log,
                    filemode='w')
logger = logging.getLogger(__name__)
def printlog(message: object, printout: object = True) -> object:
    message = '{}: {}'.format(datetime.now(), message)
    if printout:
        print(message)
    logger.info(message)

# ---------- set seed for random number ---------- 
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
setup_seed(args.seed)

# ---------- load data tsv file ---------- 
printlog('Loading data')
train_data, dev_data, test_data = load_data(args)
train_size = len(train_data)
dev_size = len(dev_data)
test_size = len(test_data)
print('Data loaded')

# ---------- calculate p, r, f1 ---------- 
def calculate(all_label_t, all_predt_t, all_clabel_t, epoch):
    exact_t = [0 for j in range(len(all_label_t))]
    for k in range(len(all_label_t)):
        if all_label_t[k] == 1 and all_label_t[k] == all_predt_t[k]:
            exact_t[k] = 1
    tpi = 0  
    li  = 0  
    pi  = 0  
    tpc = 0 
    lc  = 0  
    pc  = 0  
    for i in range(len(exact_t)):
        if exact_t[i] == 1:
            if all_clabel_t[i] == 0:
                tpi += 1
            else:
                tpc += 1
        if all_label_t[i] == 1:
            if all_clabel_t[i] == 0:
                li += 1
            else:
                lc += 1
        if all_predt_t[i] == 1:
            if all_clabel_t[i] == 0:
                pi += 1
            else:
                pc += 1
    printlog('\tINTRA-SENTENCE:')
    recli = tpi / li
    preci = tpi / (pi + 1e-9)
    f1cri = 2 * preci * recli / (preci + recli + 1e-9)
    intra = {
        'epoch': epoch,
        'p': preci,
        'r': recli,
        'f1': f1cri
    }
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpi, pi, li))
    printlog("\t\tprecision score: {}".format(intra['p']))
    printlog("\t\trecall score: {}".format(intra['r']))
    printlog("\t\tf1 score: {}".format(intra['f1']))
    reclc = tpc / lc
    precc = tpc / (pc + 1e-9)
    f1crc = 2 * precc * reclc / (precc + reclc + 1e-9)
    cross = {
        'epoch': epoch,
        'p': precc,
        'r': reclc,
        'f1': f1crc
    }
    printlog('\tCROSS-SENTENCE:')
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpc, pc, lc))
    printlog("\t\tprecision score: {}".format(cross['p']))
    printlog("\t\trecall score: {}".format(cross['r']))
    printlog("\t\tf1 score: {}".format(cross['f1']))
    return tpi + tpc, pi + pc, li + lc, intra, cross

# ---------- network ----------
if args.model == 'base':
    net = Base(args).to(device)
elif args.model == 'lstm':
    net = modified_with_lstm(args).to(device)
elif args.model == 'gat':
    net = modified_with_attention_mask(args).to(device)
else:
    print('model name error!')
    breakpoint()

optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
cross_entropy = nn.CrossEntropyLoss().to(device)

# save model and result
best_intra = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
best_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
best_intra_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
dev_best_intra_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}

best_epoch = 0
printlog('batch_size:{}'.format(args.batch_size))
printlog('num_epoch: {}'.format(args.num_epoch))
printlog('initial_lr: {}'.format(args.lr))
printlog('seed: {}'.format(args.seed))
printlog('mlp_size: {}'.format(args.mlp_size))

printlog('Start training ...')
breakout = 0

# ----------  epoch  ----------
for epoch in range(args.num_epoch):
    print('=' * 20)
    printlog('Epoch: {}'.format(epoch))
    torch.cuda.empty_cache()

    all_indices = torch.randperm(train_size).split(args.batch_size)
    loss_epoch = 0.0
    acc = 0.0

    start = time.time()

# ----------  train  ----------
    net.train()
    progress = tqdm.tqdm(total=len(train_data) // args.batch_size + 1, ncols=75,
                         desc='Train {}'.format(epoch))
    total_step = len(train_data) // args.batch_size + 1
    step = 0
    for i, batch_indices in enumerate(all_indices, 1):
        progress.update(1)
        # get a batch of wordvecs
        batch_arg, mask_arg, e_idx, label, clabel = get_batch(train_data, batch_indices)

        batch_arg = batch_arg.to(device)
        mask_arg = mask_arg.to(device)
        e_idx = e_idx.to(device)
        length = len(batch_indices)
        
        # fed data into network
        prediction = net(batch_arg, mask_arg, e_idx, length)

        predt = torch.argmax(prediction, dim=1).detach()
        label = torch.LongTensor(label).to(device)
        num_correct = (predt == label).sum()
        acc += num_correct.item()

        # loss
        loss = cross_entropy(prediction, label)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        # report
        loss_epoch += loss.item()
        if i % (3000 // args.batch_size) == 0:
            printlog('loss={:.4f}, acc={:.4f}'.format(
                loss_epoch / (3000 // args.batch_size), acc / 3000,))
            loss_epoch = 0.0
            acc = 0.0
    end = time.time()
    print('Training Time: {:.2f}s'.format(end - start))

    progress.close()

# ----------  dev  ----------
    all_indices = torch.randperm(dev_size).split(args.batch_size)
    all_label = []
    all_predt = []
    all_clabel = []

    progress = tqdm.tqdm(total=len(dev_data) // args.batch_size + 1, ncols=75,
                          desc='Eval {}'.format(epoch))

    net.eval()
    for batch_indices in all_indices:
        progress.update(1)

        # get a batch of dev_data
        batch_arg, mask_arg, e_idx, label, clabel = get_batch(dev_data, batch_indices)

        batch_arg = batch_arg.to(device)
        mask_arg = mask_arg.to(device)
        e_idx = e_idx.to(device)
        length = len(batch_indices)

        # fed data into network
        prediction = net(batch_arg, mask_arg, e_idx, length)
        predt = torch.argmax(prediction, dim=1).detach().cpu().tolist()
        all_label += label
        all_predt += predt
        all_clabel += clabel

    progress.close()

# ----------  test  ----------    
    all_indices = torch.arange(test_size).split(args.batch_size)
    all_label_t = []
    all_predt_t = []
    all_clabel_t = []
    acc = 0.0

    progress = tqdm.tqdm(total=len(test_data) // args.batch_size + 1, ncols=75,
                         desc='Eval {}'.format(epoch))

    net.eval()
    for batch_indices in all_indices:
        progress.update(1)

        # get a batch of dev_data
        batch_arg, mask_arg, e_idx, label, clabel = get_batch(test_data, batch_indices)

        batch_arg = batch_arg.to(device)
        mask_arg = mask_arg.to(device)
        e_idx = e_idx.to(device)
        length = len(batch_indices)
        
        # fed data into network
        prediction = net(batch_arg, mask_arg, e_idx, length)
        predt = torch.argmax(prediction, dim=1).detach().cpu().tolist()
        # label = torch.LongTensor(label).to(device)

        all_label_t += label
        all_predt_t += predt
        all_clabel_t += clabel    

    progress.close()
    
# ----------  report  ----------
    printlog('-------------------')

    printlog("DEV:")
    d_1, d_2, d_3, dev_intra, dev_cross = calculate(all_label, all_predt, all_clabel, epoch)
    dev_intra_cross = {
        'epoch': epoch,
        'p': precision_score(all_label, all_predt, average=None)[1],
        'r': recall_score(all_label, all_predt, average=None)[1],
        'f1': f1_score(all_label, all_predt, average=None)[1]
    }

    printlog('\tINTRA + CROSS:')
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(d_1, d_2, d_3))
    printlog("\t\tprecision score: {}".format(dev_intra_cross['p']))
    printlog("\t\trecall score: {}".format(dev_intra_cross['r']))
    printlog("\t\tf1 score: {}".format(dev_intra_cross['f1']))

    breakout += 1

# ----------  record the best result & early stop ----------
    if dev_intra_cross['f1'] > dev_best_intra_cross['f1']:
        printlog('New best epoch...')
        dev_best_intra_cross = dev_intra_cross
        best_intra_cross = dev_intra_cross
        best_intra = dev_intra
        best_cross = dev_cross
        best_epoch = epoch
        breakout = 0
        np.save('predt.npy', all_predt_t)

    printlog('=' * 20)
    printlog('Best result at epoch: {}'.format(best_epoch))
    printlog('Eval intra: {}'.format(best_intra))
    printlog('Eval cross: {}'.format(best_cross))
    printlog('Eval intra cross: {}'.format(best_intra_cross))
    printlog('Breakout: {}'.format(breakout))

    if breakout == 3:
        break