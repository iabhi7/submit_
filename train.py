import os
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data

from data_dir import AnnotationTransform, CHECKOUTDetection, input_target_stack, CHECKOUTroot, CHECKOUT_CLASSES
from layers.modules import MultiBoxLoss
from ssd_network_architecture import build_ssd
import numpy as np
import time

home = os.path.expanduser("~")
ddir = os.path.join(home,"data/dev/")

CHECKOUTroot = ddir 

SHUFFLE = True

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


basenet = vgg16_reducedfc
batch_size = 32
resume = False
num_workers = 4
iterations = 1000000
jaccard_threshold = 0.5
start_iter = 0
cuda = True
lr = 1e-3
momemtum = 0.9
weight_decay = 5e-4
gamma = 0.1
log_iters = True
save_folders = weights/
checout_root = CHECKOUTroot


torch.set_default_tensor_type('torch.cuda.FloatTensor')

cfg = {'feature_maps' : [38, 19, 10, 5, 3, 1], 'min_dim' : 300, 'steps' : [8, 16, 32, 64, 100, 300],'min_sizes' : [30, 60, 111, 162, 213, 264], 
    'max_sizes' : [60, 111, 162, 213, 264, 315], 'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]], 'variance' : [0.1, 0.2], 'clip' : True,}

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

ssd_dim = 300 
means = (104, 117, 123) 
num_classes = len(CHECKOUT_CLASSES) + 1
batch_size = batch_size
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
max_iter = 120000
weight_decay = 0.0005
stepvalues = (80000, 100000, 120000)
gamma = 0.1
momentum = 0.9

ssd_net = build_ssd('train', 300, num_classes)
net = ssd_net


net = torch.nn.DataParallel(ssd_net)
cudnn.benchmark = True

if resume:
    print('Resuming training, loading {}...'.format(resume))
    ssd_net.load_weights(resume)
else:
    vgg_weights = torch.load(save_folder + basenet)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)


net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, cuda)


def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    dataset = CHECKOUTDetection(checkout_root, AnnotationTransform())

    epoch_size = len(dataset) // batch_size
    print('Training on', dataset.name)
    step_index = 0
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=num_workers,
                                  shuffle=True, collate_fn=input_target_stack, pin_memory=True)
    for iteration in range(start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, gamma, step_index)
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        images, targets = next(batch_iterator)

        if cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if iteration % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')

        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_0712_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), save_folder + '' + version + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step"""
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
