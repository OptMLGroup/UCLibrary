############

#   @File name: synthetic_data.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-03-01 09:33:43

# @Last modified by:   Heerye
# @Last modified time: 2018-04-17T17:16:22-04:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############
import torch
from torch.autograd import Variable
import numpy

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

class SyntheticData(object):
    def __init__(self, case_idx):
        self.n_points = int(config['NN']['N_POINTS'])
        self.selectCase(case_idx)
        self.generateData()

    def to_one_hot(self, y):
        y_tensor = y.data if isinstance(y, Variable) else y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)

        return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot.type(torch.DoubleTensor)

    def generateData(self):
        loss_type = str(config['NN']['LOSS_TYPE'])
        train_ratio = float(config['NN']['TRAIN_RATIO'])

        x_axis = torch.ones(self.n_points, 1)
        y_axis = torch.ones(self.n_points, 1)

        x, y = {}, {}
        labels = []

        for idx, center_label in enumerate(self.center_label_list):

            a, b, c, d = center_label
            labels.append(c)

            std = numpy.random.random() * 0.5 + 0.5
            x[idx] = torch.normal(torch.cat((x_axis * a, y_axis * b), 1), d * std)
            y[idx] = torch.ones(self.n_points) * c

        x = torch.cat(tuple(x.values()), 0).type(torch.DoubleTensor)
        y = torch.cat(tuple(y.values()), ).type(torch.LongTensor)

        x = x.numpy()
        x = 1 - 2 * (x - x.min(0)) / (x.max(0) - x.min(0))
        x = torch.from_numpy(x).type(torch.DoubleTensor)

        loss_type = str(config['NN']['LOSS_TYPE'])
        if loss_type == 'MSELoss':
            y = to_one_hot(y)

        perm = torch.randperm(len(y))

        wall = int(train_ratio * len(y))

        self.train_x, self.train_y = x[perm[:wall]], y[perm[:wall]]
        self.test_x, self.test_y = x[perm[wall:]], y[perm[wall:]]

        self.n_features = x.size()[1]
        self.n_classes = len(set(labels))
        self.n_points = len(y)

        # return train_x, train_y, test_x, test_y, F, len(set(labels)), len(y)

    def selectCase(self, idx):
        if idx == 0:
            # (x1, x2), label, std
            self.n_classes = 2
            std = 0.8
            self.center_label_list = [[1, 1, 0, std],
                                 [2, 2, 1, std]]

        elif idx == 1:
            # (x1, x2), label, std
            self.n_classes = 4
            std = 0.5
            self.center_label_list = [[1, 1, 0, std],
                                 [2, 2, 1, std],
                                 [1, 2, 2, std],
                                 [2, 1, 3, std]]

        elif idx == 2:
            n_clusters = 12
            self.n_classes = 5
            self.center_label_list = [[]] * n_clusters
            for i in range(n_clusters):
                std = 1.0 / n_clusters
                label = numpy.random.randint(0, self.n_classes)
                x1 = numpy.random.randint(0, n_clusters) / n_clusters
                x2 = numpy.random.randint(0, n_clusters) / n_clusters
                self.center_label_list[i] = [x1, x2, label, std]

        elif idx == 3:
            n_clusters = 40
            self.n_classes = 10
            self.center_label_list = [[]] * n_clusters
            for i in range(n_clusters):
                std = 1.5 / n_clusters
                label = numpy.random.randint(0, self.n_classes)
                x1 = numpy.random.randint(0, self.n_clusters) / n_clusters
                x2 = numpy.random.randint(0, self.n_clusters) / n_clusters
                self.center_label_list[i] = [x1, x2, label, std]
        else:
            print('invalid data!!')
            return


'''
# import matplotlib
# import matplotlib.pyplot as plt

# import numpy
# from settings import args
# use_cuda = args.use_cuda

# numpy.random.seed(args.numpy_seed)
# torch.manual_seed(args.torch_seed)

def rename(s):
    return s.replace('_', '-').replace('.', '-').replace(' ', '-').replace(',', '-') + '.png'

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot.type(torch.DoubleTensor)
    # return y_one_hot

def generate_sin_data(t,left=0,right=10,i=0.01):
    x=numpy.arange(left,right,i)
    sin=numpy.sin(t*x)
    #sin=t*x
    data1=sin+0.8
    data2=sin-0.8
    data1=numpy.array([[xx,i] for i,xx in zip(data1,x)])
    data2=numpy.array([[xx,i] for i,xx in zip(data2,x)])
    data=numpy.concatenate([data1,data2])
    label=numpy.concatenate([numpy.array([1 for i in range(len(x))]) ,numpy.array([0 for i in range(len(x))])])
    idx = numpy.random.permutation(len(data))
    x_,y = data[idx], label[idx]

    x_ = 1 - 2 * (x_ - x_.min(0)) / (x_.max(0) - x_.min(0))

    traindata=torch.from_numpy(x_[:int(0.8*len(data))]).type(torch.FloatTensor)
    trainlabel=torch.from_numpy(y[:int(0.8*len(data))]).type(torch.LongTensor)
    testdata=torch.from_numpy(x_[int(0.8*len(data)):]).type(torch.FloatTensor)
    testlabel=torch.from_numpy(y[int(0.8*len(data)):]).type(torch.LongTensor)
    F=2
    BB=2
    length=len(x)
    # print (traindata[0:10],trainlabel[0:10])
    return traindata,trainlabel,testdata,testlabel,F,BB,length

def generate_data(n_points, n_classes, center_label_list):

    x_axis = torch.ones(n_points, 1)
    y_axis = torch.ones(n_points, 1)

    x, y = {}, {}
    labels = []

    for idx, center_label in enumerate(center_label_list):

        a, b, c, d = center_label
        labels.append(c)

        std = numpy.random.random() * 0.5 + 0.5
        x[idx] = torch.normal(torch.cat((x_axis * a, y_axis * b), 1), d * std)
        y[idx] = torch.ones(n_points) * c

    x = torch.cat(x.values(), 0).type(torch.DoubleTensor)
    y = torch.cat(y.values(), ).type(torch.LongTensor)
    # y = torch.cat(y.values(), ).type(torch.DoubleTensor)

    x = x.numpy()
    x = 1 - 2 * (x - x.min(0)) / (x.max(0) - x.min(0))
    x = torch.from_numpy(x).type(torch.DoubleTensor)

    # if args.lifting:
    #     tmp = [x]
    #     tmp.append((x[:, 0] + x[:, 1]).view(-1, 1))
    #     tmp.append((torch.sin(x[:, 0] + x[:, 1])).view(-1, 1))
    #     tmp.append((torch.cos(x[:, 0] + x[:, 1])).view(-1, 1))
    #     tmp.append((x[:, 0] - x[:, 1]).view(-1, 1))
    #     tmp.append((torch.sin(x[:, 0] - x[:, 1])).view(-1, 1))
    #     tmp.append((torch.cos(x[:, 0] - x[:, 1])).view(-1, 1))
    #     tmp.append((x[:, 0] * x[:, 1]).view(-1, 1))
    #     for i in range(100):
    #         tmp.append((torch.sin(x[:, 0] * x[:, 1] * i)).view(-1, 1))
    #         tmp.append((torch.cos(x[:, 0] * x[:, 1] * i)).view(-1, 1))

    #     x = torch.cat(tmp, 1)

    if args.loss == 'MSELoss':
        y = to_one_hot(y)

    perm = torch.randperm(len(y))

    wall = int(args.train_ratio * len(y))

    train_x, train_y = x[perm[:wall]], y[perm[:wall]]
    test_x, test_y = x[perm[wall:]], y[perm[wall:]]

    assert len(set(labels)) == n_classes, "labels not enough!"

    F = x.size()[1]

    return train_x, train_y, test_x, test_y, F, len(set(labels)), len(y)


def box_data(n):
    x = torch.from_numpy(
        numpy.mgrid[-1:1+1/n:1 / n, -1:1+1/n:1 / n].reshape(2, -1).T)

    if args.lifting:
        tmp = [x]
        tmp.append((x[:, 0] + x[:, 1]).view(-1, 1))
        tmp.append((torch.sin(x[:, 0] + x[:, 1])).view(-1, 1))
        tmp.append((torch.cos(x[:, 0] + x[:, 1])).view(-1, 1))
        tmp.append((x[:, 0] - x[:, 1]).view(-1, 1))
        tmp.append((torch.sin(x[:, 0] - x[:, 1])).view(-1, 1))
        tmp.append((torch.cos(x[:, 0] - x[:, 1])).view(-1, 1))
        tmp.append((x[:, 0] * x[:, 1]).view(-1, 1))
        for i in range(100):
            tmp.append((torch.sin(x[:, 0] * x[:, 1] * i)).view(-1, 1))
            tmp.append((torch.cos(x[:, 0] * x[:, 1] * i)).view(-1, 1))

        x = torch.cat(tmp, 1)
    return x


if args.idx == 0:
    # (x1, x2), label, std
    n_classes = 2
    std = 0.8
    center_label_list = [[1, 1, 0, std],
                         [2, 2, 1, std]]

elif args.idx == 1:
    # (x1, x2), label, std
    n_classes = 4
    std = 0.5
    center_label_list = [[1, 1, 0, std],
                         [2, 2, 1, std],
                         [1, 2, 2, std],
                         [2, 1, 3, std]]

elif args.idx == 2:
    n_clusters = 12
    n_classes = 5
    center_label_list = [[]] * n_clusters
    for i in range(n_clusters):
        std = 1.0 / n_clusters
        label = numpy.random.randint(0, n_classes)
        x1 = numpy.random.randint(0, n_clusters) / n_clusters
        x2 = numpy.random.randint(0, n_clusters) / n_clusters
        center_label_list[i] = [x1, x2, label, std]

elif args.idx == 3:
    n_clusters = 40
    n_classes = 10
    center_label_list = [[]] * n_clusters
    for i in range(n_clusters):
        std = 1.5 / n_clusters
        label = numpy.random.randint(0, n_classes)
        x1 = numpy.random.randint(0, n_clusters) / n_clusters
        x2 = numpy.random.randint(0, n_clusters) / n_clusters
        center_label_list[i] = [x1, x2, label, std]
else:
    print('invalid data!!')


def draw_data(x, y, t_x, t_y, F, c, n):
    fig = plt.figure(1)
    fig.subplots_adjust(bottom=0.13, left=0.16, wspace=1.)
    plt.style.use('seaborn-whitegrid')

    # plt.subplot(121)
    if use_cuda:
        plt.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[
                    :, -1], c=y.cpu().numpy(), s=50, alpha=0.3, cmap='rainbow')
    else:
        plt.scatter(x.numpy()[:, 0], x.numpy()[:, -1],
                    c=y.numpy(), s=50, alpha=0.3, cmap='rainbow')
    plt.title('Training, number of labels: %d' % c, fontsize=15)
    # plt.savefig('images/data-' + str(args.idx) +
    # '.png', format='png', dpi=1000)
    plt.show()
    # plt.close()

    fig = plt.figure(2)
    fig.subplots_adjust(bottom=0.13, left=0.16)
    plt.style.use('seaborn-whitegrid')
    # plt.subplot(122)
    if use_cuda:
        plt.scatter(t_x.cpu().numpy()[:, 0], t_x.cpu().numpy()[
                    :, -1], c=t_y.cpu().numpy(), s=50, alpha=0.3, cmap='rainbow')
    else:
        plt.scatter(t_x.numpy()[:, 0], t_x.numpy()[:, -1],
                    c=t_y.numpy(), s=50, alpha=0.3, cmap='rainbow')
    plt.title('Testing, number of labels: %d' % c, fontsize=15)
    # plt.savefig('images/tdata-' + str(args.idx) +
    # '.png', format='png', dpi=1000)
    plt.show()
    plt.close()

    return plt


def draw_box(fx, fy, x, y, c, name=''):
    fig = plt.figure(1)
    fig.subplots_adjust(bottom=0.13, left=0.16, wspace=1.)
    plt.style.use('seaborn-whitegrid')
    if use_cuda:
        plt.scatter(fx.cpu().numpy()[:, 0], fx.cpu().numpy()[
                    :, -1], c=fy, s=10, alpha=0.1, cmap='rainbow')
        plt.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[
                    :, -1], c=y.cpu().numpy(), s=50, alpha=0.9, cmap='rainbow')
    else:
        plt.scatter(fx.numpy()[:, 0], fx.numpy()[:, -1],
                    c=fy, s=10, alpha=0.1, cmap='rainbow')
        plt.scatter(x.numpy()[:, 0], x.numpy()[:, -1],
                    c=y.numpy(), s=50, alpha=0.9, cmap='rainbow')
    plt.title(name + ', training, c=%d' % c, fontsize=15)
    plt.savefig('images/' + rename('boxdata_' + str(args.idx) +
                                   '_' + name), format='png', dpi=1000)
    # plt.show()
    print('save to : %s' % rename('boxdata_' + str(args.idx) +'_' + name))
    plt.close()


if __name__ == '__main__':

    from settings import args
    import pickle

    n_points = args.n
    x, y, t_x, t_y, F, c, n = generate_data(
        n_points, n_classes, center_label_list)

    x, y, t_x, t_y, F, c, n = generate_sin_data(2, 0, 10, 0.1)

    print(F, c, n)

    draw_data(x, y, t_x, t_y, F, c, n)
    # save_name = 'easy_data.pkl'
    #
    # with open(save_name, 'wb') as f:
    #     pickle.dump([x.numpy(), y.numpy(), t_x.numpy(), t_y.numpy(), F, c, n], f)
    #
    # with open(save_name, 'rb') as f:
    #     data = pickle.load(f)
    # X, Y, t_X, t_Y, F, C, n = data
    # draw_data(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(t_X), torch.from_numpy(t_Y), F, C, n)
'''