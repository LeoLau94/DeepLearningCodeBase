import torch
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn as nn
from torch.autograd import Variable
__all__ = ['Varying_BatchNorm1D', 'Varying_BatchNorm2D']

class Varying_BatchNorm(_BatchNorm):
    def __init__(self,*args,**kwargs):
        super(Varying_BatchNorm,self).__init__(*args,**kwargs)
        self.register_buffer('varying_lambda',torch.zeros(self.num_features))
        self.register_buffer('avg_values',self.weight.data)
        self.register_buffer('lambda_delta',torch.zeros(self.num_features))
        self.iteration = 1

    def updateFactors_(self,spares_rate,penalty):
        self.iteration += 1
        self.avg_values = (self.avg_values * (self.iteration-1) + self.weight.data.abs()) / self.iteration
        remains = round(spares_rate * self.num_features) - 1
        previous = round((1 - spares_rate) * self.num_features) - 1
        gamma_sorted, gamma_sorted_idx = torch.sort(self.weight.data.abs())
        threshold = gamma_sorted[remains]
        pre_threshold = gamma_sorted[previous]

        le_threshold_index_list = gamma_sorted_idx[:remains + 1]
        gt_threshold_index_list = gamma_sorted_idx[remains + 1:]
        
        self.lambda_delta[le_threshold_index_list] = penalty * (1 - self.avg_values[le_threshold_index_list] / threshold)

        self.lambda_delta[gt_threshold_index_list] = penalty * (threshold - self.avg_values[gt_threshold_index_list]) / pre_threshold
        
        self.varying_lambda.add_(self.lambda_delta)
        self.varying_lambda.masked_fill_(self.varying_lambda.lt(0),0)

    def updateGrad(self):
        self.weight.grad.data.add_(self.varying_lambda * self.weight.data)


class Varying_BatchNorm1D(Varying_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(Varying_BatchNorm1D, self)._check_input_dim(input)


class Varying_BatchNorm2D(Varying_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(Varying_BatchNorm2D, self)._check_input_dim(input)


# class net(nn.Module):
#     def __init__(self):
#         super(net,self).__init__()
#         self.conv_net = nn.Sequential(
#             nn.Conv2d(3,64,3,padding=1),
#             Varying_BatchNorm2D(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64,128,3,padding=1),
#             Varying_BatchNorm2D(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128,256,3,padding=1),
#             Varying_BatchNorm2D(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=8, stride=1),
#             )
#         self.classfier = nn.Sequential(
#             nn.Linear(256,512),
#             nn.Dropout(0.3),
#             nn.Linear(512,10)
#             )

#     def forward(self,x):
#         x = self.conv_net(x)
#         x = x.view(x.size(0),-1)
#         x = self.classfier(x)
#         return x

# def accuracy(output, target, topk=(1,)):
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

if __name__ == '__main__':
    
    net1 = net()

    x = Variable(torch.FloatTensor(16,3,32,32))
    y = Variable(torch.FloatTensor(16,10))
    opitimizer = torch.optim.SGD(net1.parameters(),lr=0.01,weight_decay=0.0001,momentum=0.9,nesterov=True)
    epoch = 100
    criterion = nn.MSELoss()

    for  e in range(epoch):
        output = net1(x)
        loss = criterion(output,y)
        opitimizer.zero_grad()
        loss.backward()
        opitimizer.step()



        for m in net1.modules():
            if isinstance(m,Varying_BatchNorm2D):
                m.updateFactors_(0.5)
                m.updateGrad()
        
        if (e + 1) % 10 == 0:
            print(loss.data[0])
            for (i,m) in enumerate(net1.modules()):
                if isinstance(m,Varying_BatchNorm2D):
                    print(i)
                    print(m.weight.data)

    # print(y.size())
