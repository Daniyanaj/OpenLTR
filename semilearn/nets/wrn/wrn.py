import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ortho_group
from semilearn.nets.utils import load_checkpoint
import numpy as np

momentum = 0.001


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, **kwargs):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        #self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]
        self.num_features = channels[3]

        #self.num_classes = num_classes
        
        # Create an orthogonal matrix and convert it to torch tensor
        orth = ortho_group.rvs(dim=512).astype(np.float32)
        orth = torch.tensor(orth[:, :num_classes])
        
        # Calculate ETF (Eigen Transformation Factor)
        etf = torch.sqrt(torch.tensor(num_classes / (num_classes - 1.0))) * \
              orth @ (torch.eye(num_classes) - 1.0 / num_classes * torch.ones(num_classes, num_classes))
        etf = etf.t()  # Transpose the ETF matrix
        
        # Initialize the classifier layer with ETF
        self.fc = nn.Sequential(
            nn.Linear(channels[3], num_classes),
            nn.ReLU(inplace=True)
        )
        
        # Set the weights and biases of the classifier layer
        self.fc[0].weight.data = etf / etf.norm(dim=-1, keepdim=True)
        self.fc[0].bias.data = torch.zeros(self.num_classes)
        
        # Flag to indicate ETF initialization
        #self.etf = True

        # rot_classifier for Remix Match
        # self.is_remix = is_remix
        # if is_remix:
        #     self.rot_classifier = nn.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        #print(x.shape)
        if only_fc:
            return self.fc(x)
        
        out = self.extract(x)
        #print(out.shape, "ada0")
        out = F.adaptive_avg_pool2d(out, 1)
        #print(out.shape, "ada")
        out = out.view(-1, self.channels)
       

        if only_feat:
            return out
        
        output = self.fc(out)
        result_dict = {'logits':output, 'feat':out}
        return result_dict

    def extract(self, x):
        #print(x.shape,"1")
        out = self.conv1(x)
        #print(out.shape,"2")
        out = self.block1(out)
        #print(out.shape,"3")
        out = self.block2(out)
        #print(out.shape,"4")
        out = self.block3(out)
        #print(out.shape,"5")
        out = self.relu(self.bn1(out))
        #print(out.shape,"6")
        return out

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}conv1'.format(prefix), blocks=r'^{}block(\d+)'.format(prefix) if coarse else r'^{}block(\d+)\.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd


def wrn_28_2(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=28, widen_factor=2, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def wrn_28_8(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=28, widen_factor=8, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model

def wrn_16_2(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=16, widen_factor=2, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model
def wrn_50_2(pretrained=False, pretrained_path=None, **kwargs):
    model = WideResNet(first_stride=1, depth=50, widen_factor=2, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model




if __name__ == '__main__':
    model = wrn_16_2(pretrained=True, num_classes=10)
