# -*- coding: utf-8 -*-

import sys
import os
#sys.path.append(os.getcwd())
os.system('dir')
os.system('ls')
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils,nn
#import gluoncv
import time
import d2lzh as d2l
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
from gluoncv import model_zoo


def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
        
def plot(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(7, 5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.plot(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.plot(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)

############################################################################
#loss function
#loss = gloss.SoftmaxCrossEntropyLoss()

def VoteLoss(y_hat, y, v=5):
    p = nd.pick(y_hat, y)
    loss = nd.square(p-1)
    #loss = v*p
    return nd.sum(loss, axis=0)


def AugFocalLoss(y_hat, y, gamma=1):
    y_hat = nd.softmax(y_hat)
    ctx = y_hat.ctx
    alpha = nd.array([0.053, 0.211, 0.086, 0.049, 0.158, 0.092, 0.258, 0.094],ctx=ctx) + nd.zeros(shape=(y.size, 1),ctx=ctx)
    loss1 = gloss.SoftmaxCrossEntropyLoss()
    #Fl =     -alpha       *         (1-p)**gamma        *  ln(p) 
    fl = nd.pick(alpha, y) * ((1-nd.pick(y_hat, y))**gamma) * loss1(y_hat, y)
    return fl


def loss(y_hat, y, alpha=0):
    vote_l = VoteLoss(y_hat, y)
    loss1 = gloss.SoftmaxCrossEntropyLoss()
    entropy_l = loss1(y_hat, y)
    return entropy_l + alpha*vote_l 

############################################################################

def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])
############################################################################
def train(net, train_iter, test_iter, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    print('training on', ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    train_acc, test_acc = [], []
    train_loss = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        duandian = lr_period *2 + 1
        if epoch>0 and epoch % lr_period == 0 and epoch < duandian:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        #超过40轮每10轮学习率自乘lr_decay
        if epoch>duandian and epoch % 10 == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [AugFocalLoss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                 for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        time_s = "time %.2f sec" % (time.time() - start)
        train_acc.append(train_acc_sum/n)
        train_loss.append(train_l_sum/n)
        if test_iter is not None:
            test_accu = d2l.evaluate_accuracy(test_iter, net, ctx)
            epoch_s = ("epoch %d, loss %f, train acc %f, test_acc %f, "
                       % (epoch + 1, train_l_sum / n, train_acc_sum / n,
                          test_accu))
            test_acc.append(test_accu)
        else:
            epoch_s = ("epoch %d, loss %f, train acc %f, " %
                       (epoch + 1, train_l_sum / n, train_acc_sum / n))
        print(epoch_s + time_s + ', lr ' + str(trainer.learning_rate))
    plot(range(1, num_epochs + 1), train_acc, 'epochs', 'accuracy',
              range(1, num_epochs + 1), test_acc, ['train', 'test'])
    plt.figure()
    plot(range(1, num_epochs + 1), train_loss, 'epochs', 'loss', legend=['loss'])