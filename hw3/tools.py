import itertools, imageio, torch
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets
from scipy.misc import imresize
from torch.autograd import Variable
from IPython.display import clear_output


def show_result(G, x_, y_, num_epoch, show = False, save = False, path = 'result.png'):
    test_images = G(x_)
    size_figure_grid = 3
    fig, ax = plt.subplots(x_.size()[0], size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(x_.size()[0]), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for i in range(x_.size()[0]):
        ax[i, 0].cla()
        ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 1].cla()
        ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 2].cla()
        ax[i, 2].imshow((y_[i].numpy().transpose(1, 2, 0) + 1) / 2)
    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist):
    x = range(len(hist['dis_loss']))
    y1 = hist['dis_loss']
    y2 = hist['gen_loss']
    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)


def data_load(path, subfolder, transform, batch_size, shuffle=True):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]
    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1
        n += 1
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)


def imgs_resize(imgs, resize_scale = 286):
    outputs = torch.FloatTensor(imgs.size()[0], imgs.size()[1], resize_scale, resize_scale)
    for i in range(imgs.size()[0]):
        img = imresize(imgs[i].numpy(), [resize_scale, resize_scale])
        outputs[i] = torch.FloatTensor((img.transpose(2, 0, 1).astype(np.float32).reshape(-1, imgs.size()[1], resize_scale, resize_scale) - 127.5) / 127.5)
    return outputs


def random_crop(imgs1, imgs2, crop_size = 256):
    outputs1 = torch.FloatTensor(imgs1.size()[0], imgs1.size()[1], crop_size, crop_size)
    outputs2 = torch.FloatTensor(imgs2.size()[0], imgs2.size()[1], crop_size, crop_size)
    for i in range(imgs1.size()[0]):
        img1 = imgs1[i]
        img2 = imgs2[i]
        rand1 = np.random.randint(0, imgs1.size()[2] - crop_size)
        rand2 = np.random.randint(0, imgs2.size()[2] - crop_size)
        outputs1[i] = img1[:, rand1: crop_size + rand1, rand2: crop_size + rand2]
        outputs2[i] = img2[:, rand1: crop_size + rand1, rand2: crop_size + rand2]
    return outputs1, outputs2


def random_fliplr(imgs1, imgs2):
    outputs1 = torch.FloatTensor(imgs1.size())
    outputs2 = torch.FloatTensor(imgs2.size())
    for i in range(imgs1.size()[0]):
        if torch.rand(1)[0] < 0.5:
            img1 = torch.FloatTensor(
                (np.fliplr(imgs1[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs1.size()[1], imgs1.size()[2], imgs1.size()[3]) + 1) / 2)
            outputs1[i] = (img1 - 0.5) / 0.5
            img2 = torch.FloatTensor(
                (np.fliplr(imgs2[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs2.size()[1], imgs2.size()[2], imgs2.size()[3]) + 1) / 2)
            outputs2[i] = (img2 - 0.5) / 0.5
        else:
            outputs1[i] = imgs1[i]
            outputs2[i] = imgs2[i]
    return outputs1, outputs2


def show_results(n_pic=5):
    curr_dir = os.path.abspath(os.getcwd())
    img_list = sorted(os.listdir(os.path.join(curr_dir, 'facades_results/test_results')))[-n_pic * 3 - 2: -2]
    filt_img = []
    for img in img_list:    
        pat = img.split('_')[1].split('.')[0]
        if (pat == 'target') or (pat == 'output'):
            filt_img.append(img)
    nrows = n_pic
    ncols = 2
    figsize = (n_pic * 2, n_pic * 3)
    cols = ['Generated picture', 'Input picture']
    fig, ax = plt.subplots(nrows=nrows,
                           ncols=ncols,
                           figsize=figsize,
                           )
    for i, axi in enumerate(ax.flat):      
        img = plt.imread('facades_results/test_results/' + filt_img[i])
        axi.imshow(img)
        axi.axis('off')
    for ax, col in zip(ax[0], cols):
        ax.set_title(col, fontdict={'fontsize': 16})
    plt.tight_layout(True)
    plt.show()


def train_routine(train_epoch, train_loader, img_size, input_size, resize_scale, crop_size,
                  random_fliplr, device, BCE_loss, L1_loss, G_optimizer, D_optimizer, D, G):
    epoch_ = []
    d_loss_ = []
    g_loss_ = []
    for epoch in tqdm.tqdm_notebook(range(train_epoch)):
        D_losses = []
        G_losses = []
        num_iter = 0
        for x_, _ in train_loader:
            D.zero_grad()
            y_ = x_[:, :, :, 0:img_size]
            x_ = x_[:, :, :, img_size:]
            if img_size != input_size:
                x_ = imgs_resize(x_, input_size)
                y_ = imgs_resize(y_, input_size)
            if resize_scale:
                x_ = imgs_resize(x_, resize_scale)
                y_ = imgs_resize(y_, resize_scale)
            if crop_size:
                x_, y_ = random_crop(x_, y_, crop_size)
            x_, y_ = random_fliplr(x_, y_)
            x_, y_ = Variable(x_.to(device)), Variable(y_.to(device))
            D_result = D(x_, y_).squeeze()
            D_real_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).to(device)))
            G_result = G(x_)
            D_result = D(x_, G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, Variable(torch.zeros(D_result.size()).to(device)))
            D_train_loss = (D_real_loss + D_fake_loss) * 0.5
            D_train_loss.backward()
            D_optimizer.step()
            D_losses.append(D_train_loss.data)
            G.zero_grad()
            G_result = G(x_)
            D_result = D(x_, G_result).squeeze()
            G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).to(device))) + 100 * L1_loss(G_result, y_)
            G_train_loss.backward()
            G_optimizer.step()
            G_losses.append(G_train_loss.data)
            num_iter += 1
        epoch_.append(epoch+1)
        d_loss_.append(torch.mean(torch.FloatTensor(D_losses)))
        g_loss_.append(torch.mean(torch.FloatTensor(G_losses)))
        clear_output(True) 
        print('Epoch [%d/%d]' % ((epoch + 1), train_epoch))
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
        ax[0].set_title('Discriminator loss', fontsize=14)
        ax[0].plot(epoch_, d_loss_, 'y')
        ax[0].grid()
        ax[1].set_title('Generator loss', fontsize=14)
        ax[1].plot(epoch_, g_loss_, 'b')
        ax[1].grid()
        plt.show()


def inference(test_loader, device, G, dataset, input_size):
    n = 0
    for x_, _ in test_loader:
        y_ = x_[:, :, :, :x_.size()[2]]
        x_ = x_[:, :, :, x_.size()[2]:]
        if x_.size()[2] != input_size:
            x_ = imgs_resize(x_, input_size)
            y_ = imgs_resize(y_, input_size)
        x_ = Variable(x_.to(device), volatile=True)
        test_image = G(x_)
        s = test_loader.dataset.imgs[n][0][::-1]
        s_ind = len(s) - s.find('/')
        e_ind = len(s) - s.find('.')
        ind = test_loader.dataset.imgs[n][0][s_ind:e_ind-1]
        path = dataset + '_results/test_results/' + ind + '_input.png'
        plt.imsave(path, (x_[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        path = dataset + '_results/test_results/' + ind + '_output.png'
        plt.imsave(path, (test_image[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        path = dataset + '_results/test_results/' + ind + '_target.png'
        plt.imsave(path, (y_[0].numpy().transpose(1, 2, 0) + 1) / 2)
        n += 1