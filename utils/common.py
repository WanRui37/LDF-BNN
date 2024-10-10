from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import os
import math
import torch
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.args = args
        self.job_dir = Path(args.job_dir)
        self.ckpt_dir = self.job_dir / 'checkpoint'
        self.run_dir = self.job_dir / 'run'

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)

        config_dir = self.job_dir / 'config.txt'
        with open(config_dir, 'w') as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save_model(self, state, is_best):
        save_path = f'{self.ckpt_dir}/model_checkpoint.pt'
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, f'{self.ckpt_dir}/model_best.pt')


def get_logger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.

    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, model_output, real_output):

        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        
        # if real_output.requires_grad:
        #     raise ValueError("real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output
        del real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        # model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss

def get_pruning_rate(pruning_rate):
    import re

    cprate_str = pruning_rate
    print(cprate_str)
    cprate_str_list = cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate = []
    for x in cprate_str_list:
        num = 1
        find_num = re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace('*', ''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])] * num

    return cprate

def get_params_model(model):
    params = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            params.append(m.weight.data)
            if m.bias is not None:
                params.append(m.bias.data)
        elif isinstance(m, nn.Linear):
            params.append(m.weight.data)
            if m.bias is not None:
                params.append(m.bias.data)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            params.append(m.running_mean.data)
            params.append(m.running_var.data)
            if m.weight is not None:
                params.append(m.weight.data)
            if m.bias is not None:
                params.append(m.bias.data)

    return params

def get_params_model_dict(model):
    params = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            params[name + ".weight"] = m.weight.data
            if m.bias is not None:
                params[name + ".bias"] = m.bias.data
        elif isinstance(m, nn.Linear):
            params[name + ".weight"] = m.weight.data
            if m.bias is not None:
                params[name + ".bias"] = m.bias.data
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            params[name + ".running_mean"] = m.running_mean.data
            params[name + ".running_var"] = m.running_var.data
            if m.weight is not None:
                params[name + ".weight"] = m.weight.data
            if m.bias is not None:
                params[name + ".bias"] = m.bias.data
    for k in params.keys():
        print(k)
    return params

def load_params_model(model,params):
    cnt = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = params[cnt]
            cnt += 1
            if m.bias is not None:
                m.bias.data = params[cnt]
                cnt += 1
        elif isinstance(m, nn.Linear):
            m.weight.data = params[cnt]
            cnt += 1
            if m.bias is not None:
                m.bias.data = params[cnt]
                cnt += 1
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.data = params[cnt]
            cnt += 1
            m.running_var.data = params[cnt]
            cnt += 1
            if m.weight is not None:
                m.weight.data = params[cnt]
                cnt += 1
            if m.bias is not None:
                m.bias.data = params[cnt]
                cnt += 1
    return model

def convert_keys(model, baseline):
    '''
    rename the baseline's key to model's name
    e.g.
        baseline_ckpt = torch.load(args.baseline, map_location=device)
        model.load_state_dict(convert_keys(model, baseline_ckpt))
    '''
    from collections import OrderedDict

    baseline_state_dict = OrderedDict()
    model_key = list(model.state_dict().keys())

    if 'model' in baseline:
        baseline = baseline['model']
    if 'state_dict' in baseline:
        baseline = baseline['state_dict']
    baseline_key = list(baseline.keys())
    if(len(model_key)!=len(baseline_key)):
        print("ERROR: The model and the baseline DO NOT MATCH")
        exit()
    else:
        for i in range(len(model_key)):
            baseline_state_dict[model_key[i]] = baseline[baseline_key[i]]
    return baseline_state_dict

def merge_ReActNet_branches(ckpt):
    import re
    from   collections import OrderedDict
    '''
    merge two branches of the ReActNet.
    args: ckpt is the checkpoint of the original ReActNet.
    '''
    ckpt_key = list(ckpt['state_dict'].keys())
    my_reactnet_state_dict = OrderedDict()

    for i in range(len(ckpt_key)):
        if( re.search('feature.\d*.binary_pw_down2',ckpt_key[i]) or re.search('feature.\d*.bn2_2',ckpt_key[i]) ):
            continue
        elif( re.search('feature.\d*.binary_pw_down1',ckpt_key[i]) ): #merge 2 conv
            my_reactnet_state_dict[ckpt_key[i]] = torch.cat( (ckpt['state_dict'][ckpt_key[i]], ckpt['state_dict'][ckpt_key[i][:-9]+'2.weights']), dim=0 )
        elif( re.search('feature.\d*.bn2_1',ckpt_key[i]) and not(re.search('num_batches_tracked',ckpt_key[i])) ): #merge 2 BN and skip the num_batches_tracked parameter
            aa = re.search('feature.\d*.bn2_1.',ckpt_key[i])
            index = aa.span()[1] #the end index of the ''feature.1.bn2_1.''  for example 'module.feature.1.bn2_1.running_var' index=23
            my_reactnet_state_dict[ckpt_key[i]] = torch.cat( (ckpt['state_dict'][ckpt_key[i]], ckpt['state_dict'][ckpt_key[i][:index-6]+'bn2_2.'+ckpt_key[i][index:]]), dim=0 )
        elif( not(re.search('bn2_2.num_batches_tracked',ckpt_key[i])) ): #copy all the rest parameters
            my_reactnet_state_dict[ckpt_key[i]] = ckpt['state_dict'][ckpt_key[i]]
    
    return {'state_dict':my_reactnet_state_dict}

def tensor2mat(f_name,val_name,val):
    '''
    Args: 
        f_name: the .mat file name
        val_name: the variable name in the .mat file
        val: the val to store
    '''
    val_dic=dict()

    if(len(np.shape(val))==1):
        val_dic[val_name]=val.cpu().reshape(-1,1).permute(0,1).numpy()

    if(len(np.shape(val))==2):
        val_dic[val_name]=val.cpu().permute(0,1).numpy()       

    if(len(np.shape(val))==3):
        val_dic[val_name]=val.cpu().permute(2,1,0).numpy()

    if(len(np.shape(val))==4):
        val_dic[val_name]=val.cpu().permute(3,2,1,0).numpy()

    sio.savemat(f_name,mdict=val_dic)

def const(num=0.5):
    return num


def linear(epoch,EP):
    return 1 / (EP - 1) * (epoch - 1)


def log(epoch,EP):
    return math.log(epoch+1, EP)


def exp(epoch,EP):
    return 2 ** (epoch / 5) / (2 ** ((EP / 5) - 1)) / 2


def step(epoch,EP):
    if epoch < EP / 4:
        return 0
    if epoch < EP / 2:
        return 1 / 3
    if epoch < EP * 3 / 4:
        return 2 / 3
    else:
        return 1

def sig(epoch,EP):
    scale = 5
    return 1 / (1 + np.exp(-(epoch / scale - EP / (scale * 2))))
