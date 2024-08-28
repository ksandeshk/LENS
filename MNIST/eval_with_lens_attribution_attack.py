from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import numpy as np
import shutil
import json
import math
import os
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model import Model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Global constants
with open('config.json') as config_file:
    config = json.load(config_file)

num_eval_examples = config['num_eval_examples']
epsilon = config['epsilon']
random_seed = config['random_seed']
model_dir = config['model_dir']
num_IG_steps = config['num_IG_steps']
k_top = config['k_top']
eval_k_top = config['eval_k_top']
saliency_type = config['saliency_type']
attribution_attack_method = config['attribution_attack_method']
attribution_attack_measure = config['attribution_attack_measure']
attribution_attack_step_size = config['attribution_attack_step_size']
attribution_attack_steps = config['attribution_attack_steps']
attribution_attack_times = config['attribution_attack_times']

# Obtains pixel indices around index given by x of window size 3x3
def submat(x):
        x1 = x + 28
        x2 = x - 28
        return [x2-1,x2,x2+1,x-1,x,x+1,x1-1,x1,x1+1]

# Add pixels of submat size around the pixel of index pixInd
def addwindowPx(pixInd):
    finalo = []
    finalo = finalo + submat(pixInd)

    finalo = np.array(finalo)
    finalo= finalo[finalo < imagedimflat]
    finalo= finalo[finalo > 0]
    fo = np.unique(np.array(finalo))
    return fo

# Add pixels of submat size around the list of pixel indices in inpList
def addwindowList(inpList):
    lfinalo = []
    for i in inpList:
        lfinalo = lfinalo + submat(i)

    lfinalo = np.array(lfinalo)
    lfinalo = lfinalo[lfinalo < imagedimflat]
    lfinalo = lfinalo[lfinalo > 0]
    lfo = np.unique(np.array(lfinalo))
    return lfo

# Obtain top-k-div k=eval_topk
def topkDiv(oList, eval_topk): #oList is sorted in descending order
    finalo = []
    finalo.append(oList[0])
    oList = np.setdiff1d(oList, addwindowPx(oList[0]), assume_unique=True)
    for i in range(eval_topk-1):
        oList = np.setdiff1d(oList, addwindowPx(oList[0]), assume_unique=True) 
        oList = oList[oList < imagedimflat]
        oList = oList[oList > 0]
        finalo.append(oList[0])
    finalo = np.array(finalo)
    finalo = finalo[finalo < imagedimflat]
    finalo = finalo[finalo > 0]
    fo = np.unique(np.array(finalo))
    return fo

## Attack with LENS
#if saliency_type == 'ig':
#    from ig_attack_with_lens import IntegratedGradientsAttack as SaliencyAttack
# Attack with WITHOUT LENS
if saliency_type == 'ig':
    from ig_attack import IntegratedGradientsAttack as SaliencyAttack
elif saliency_type == 'simple_gradient':
    from simple_gradient_attack import SimpleGradientAttack as SaliencyAttack
else:
    assert False, ('Unknown saliency type.')

np.random.seed(random_seed)
tf.set_random_seed(random_seed)

# Set up the data, hyperparameters, and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

reference_image = np.zeros((28,28,1))

model = Model(create_saliency_op = saliency_type)

saver = tf.train.Saver()

global_step = tf.contrib.framework.get_or_create_global_step()

checkpoint = tf.train.latest_checkpoint(model_dir)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

with tf.Session(config = tf_config) as sess:
    # Restore the checkpoint
    saver.restore(sess, checkpoint)

    test_images = mnist.test.images
    test_labels = mnist.test.labels
    
    min_intersections = []
    min_spearmans = []
    min_kendalls = []
    
    correct_cnt = 0
    sum_o_p = 0.0
    sum_fo_p = 0.0
    sum_o_fp = 0.0
    sum_fo_fp = 0.0
    sum_lfo_fp = 0.0
    sum_fo_lfp = 0.0

    n_zeros_orig_ig_test = np.empty((0,784))
    n_zeros_pert_ig_test = np.empty((0,784))
    n_zeros_image_test = np.zeros((0,784))
    n_zeros_image_test_pert = np.zeros((0,784))
    n_zeros_pert = np.zeros((0,784))
    n_zeros_image_label = []
    for i in range(num_eval_examples):
        test_image = test_images[i]
        n_zeros_image_test = np.concatenate((n_zeros_image_test,test_image.reshape([-1,784])))
        test_image = test_image.reshape([28, 28, 1])
        original_label = test_labels[i]
        n_zeros_image_label = np.append(n_zeros_image_label,original_label)

        module = SaliencyAttack(sess = sess, test_image = test_image, original_label = original_label, NET = model,
                                           attack_method = attribution_attack_method, epsilon = epsilon,
                                           k_top = k_top, eval_k_top = eval_k_top, num_steps = num_IG_steps,
                                           attack_iters = attribution_attack_steps,
                                           attack_times = attribution_attack_times,
                                           alpha = attribution_attack_step_size,  attack_measure = attribution_attack_measure,
                                           reference_image = reference_image, same_label = True)

        if module.status == 1:
            n_zeros_image_test = np.concatenate((n_zeros_image_test,test_image.reshape([-1,784])))
            n_zeros_image_label = np.append(n_zeros_image_label,original_label)
            
            correct_cnt += 1
            
            intersections, spearmans, kendalls, orig_ig, pert_ig, test_image_pert, pert = module.iterative_attack()
            #intersections, spearmans, kendalls = module.iterative_attack()
            
            idx = np.argmin(kendalls)
            min_intersections.append(intersections[idx])
            min_spearmans.append(spearmans[idx])
            min_kendalls.append(kendalls[idx])
            
            n_zeros_orig_ig_test = np.concatenate((n_zeros_orig_ig_test,np.reshape(orig_ig[idx],(1,784))))
            n_zeros_pert_ig_test = np.concatenate((n_zeros_pert_ig_test,np.reshape(pert_ig[idx],(1,784))))
            n_zeros_image_test_pert = np.concatenate((n_zeros_image_test_pert,np.reshape(test_image_pert[idx],(1,784))))
            n_zeros_pert = np.concatenate((n_zeros_pert,np.reshape(pert[idx],(1,784))))
            res_str = '{} {} '.format(i, 1)

            for k in range(attribution_attack_times):
                res_str += '{:.6f} {:.6f} {:.6f} '.format(intersections[k], spearmans[k], kendalls[k])
                

            print('progress: {}/{}, {}'.format(i + 1, num_eval_examples, res_str))
            ## LENS metric start
            eval_topk = 100
            eval_topk1 = 600 # heuristic value set for top-k-div
            imagedimflat = 784 #28*28 image size

            o = np.argsort(np.reshape(orig_ig[idx],(imagedimflat)))[-eval_topk:] # top-k
            p = np.argsort(np.reshape(pert_ig[idx],(imagedimflat)))[-eval_topk:] # top-k

            o1 = np.argsort(np.reshape(orig_ig[idx],(imagedimflat)))[-eval_topk1:][::-1] # used for top-k-div
            p1 = np.argsort(np.reshape(pert_ig[idx],(imagedimflat)))[-eval_topk1:][::-1] # used for top-k-div

            ofo = addwindowList(o) # w-LENS-recall(top-k)
            ofp = addwindowList(p) # w-LENS-prec(top-k)

            fo = topkDiv(o1, eval_topk) # top-k-div of map of unperturbed image
            fp = topkDiv(p1, eval_topk) # top-k-div of map of perturbed image

            lfo = addwindowList(fo) # w-LENS-recall(top-k-div)
            lfp = addwindowList(fp) # w-LENS-prec(top-k-div)

            sum_o_p = sum_o_p + float(len(np.intersect1d(o,p)))/eval_topk 
            sum_fo_p = sum_fo_p + float(len(np.intersect1d(ofo,p)))/eval_topk 
            sum_o_fp = sum_o_fp + float(len(np.intersect1d(o,ofp)))/eval_topk 
            sum_fo_fp = sum_fo_fp + float(len(np.intersect1d(fo,fp)))/eval_topk 
            sum_lfo_fp = sum_lfo_fp + float(len(np.intersect1d(lfo,fp)))/eval_topk 
            sum_fo_lfp = sum_fo_lfp + float(len(np.intersect1d(fo,lfp)))/eval_topk             
            ## LENS metric end
        else:
            res_str = '{} {} '.format(i, 0)

            for k in range(attribution_attack_times):
                res_str += '{:.6f} {:.6f} {:.6f} '.format(0, 0, 0)

            print('progress: {}/{}, prediction incorrect!'.format(i + 1, num_eval_examples))
            
avg_intersection = np.mean(min_intersections)
avg_spearman = np.mean(min_spearmans)
avg_kendall = np.mean(min_kendalls)

print('process {} examples'.format(num_eval_examples))
print('accuracy {}'.format(float(correct_cnt)/num_eval_examples))
print('Average top-k intersection: {:.4f}'.format(avg_intersection))
print('Average spearman rank correlation: {:.4f}'.format(avg_spearman))
print('Average kendall rank correlation: {:.4f}'.format(avg_kendall))
print("top-k intersection : ",sum_o_p/num_samples)
print("LENS-recall : ",sum_fo_p/num_samples)
print("LENS-prec   : ",sum_o_fp/num_samples)
print("top-k-div intersection : ",sum_fo_fp/num_samples)
print("LENS-recall-div : ",sum_lfo_fp/num_samples)
print("LENS-prec-div   : ",sum_fo_lfp/num_samples)
