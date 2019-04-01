from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torchvision.utils import make_grid
import math
from utils import resize_image


#################################################################################
# You will need to fill in the missing code in this file
#################################################################################


#################################################################################
# Part I: Understanding Convolutions
#################################################################################
class CustomConv2DFunction(Function):

  @staticmethod
  def forward(ctx, input_feats, weight, bias, stride=1, padding=0):
    """
    Forward propagation of convolution operation.
    We only consider square filters with equal stride/padding in width and height!

    Args:
      input_feats: input feature map of size N * C_i * H * W
      weight: filter weight of size C_o * C_i * K * K
      bias: (optional) filter bias of size C_o
      stride: (int, optional) stride for the convolution. Default: 1
      padding: (int, optional) Zero-padding added to both sides of the input. Default: 0

    Outputs:
      output: responses of the convolution  w*x+b

    """
    # sanity check
    assert weight.size(2) == weight.size(3)
    assert input_feats.size(1) == weight.size(1)
    assert isinstance(stride, int) and (stride > 0)
    assert isinstance(padding, int) and (padding >= 0)

    # save the conv params
    kernel_size = weight.size(2)
    ctx.stride = stride
    ctx.padding = padding
    ctx.input_height = input_feats.size(2)
    ctx.input_width = input_feats.size(3)

    # make sure this is a valid convolution
    assert kernel_size <= (input_feats.size(2) + 2 * padding)
    assert kernel_size <= (input_feats.size(3) + 2 * padding)

    #################################################################################
    # Fill in the code here
    #################################################################################
    # Unfold input and weight
    input_unf = unfold(input_feats, kernel_size=(kernel_size, kernel_size), padding=padding, stride=stride)
    wgt_unf = unfold(weight, kernel_size=(kernel_size, kernel_size))

    # Calculate unfolded output
    out_unf = input_unf.transpose(1, 2).matmul(wgt_unf[0:, 0:, 0].t()).transpose(1, 2).add(bias.reshape(-1,1))

    # Fold output    
    h_o = int(math.floor((input_feats.size(2) + 2 * padding - kernel_size) / stride) + 1)
    w_o = int(math.floor((input_feats.size(3) + 2 * padding - kernel_size) / stride) + 1)
    output = fold(out_unf, output_size=(h_o, w_o), kernel_size=(1, 1))

    # save for backward (you need to save the unfolded tensor into ctx)
    # ctx.save_for_backward(your_vars, weight, bias)   
    ctx.save_for_backward(input_unf, wgt_unf, weight, bias)
       
    return output

  @staticmethod
  def backward(ctx, grad_output):
    """
    Backward propagation of convolution operation

    Args:
      grad_output: gradients of the outputs

    Outputs:
      grad_input: gradients of the input features
      grad_weight: gradients of the convolution weight
      grad_bias: gradients of the bias term

    """
    # unpack tensors and initialize the grads
    # your_vars, weight, bias = ctx.saved_tensors
    input_unf, wgt_unf, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None

    # recover the conv params
    kernel_size = weight.size(2)
    stride = ctx.stride
    padding = ctx.padding
    input_height = ctx.input_height
    input_width = ctx.input_width

    #################################################################################
    # Fill in the code here
    #################################################################################
    # compute the gradients w.r.t. input and params
    # Unfold output gradient
    grad_output_unf = unfold(grad_output, kernel_size=(1, 1))

    # Calculate input gradient
    grad_input_unf = grad_output_unf.transpose(1, 2).matmul(wgt_unf[0:, 0:, 0]).transpose(1, 2)
    
    # Fold input gradient
    grad_input = fold(grad_input_unf, output_size=(input_height, input_width), kernel_size=(kernel_size, kernel_size), padding=padding, stride=stride)
    
    # Calculate weight gradient
    grad_weight_unf = torch.bmm(grad_output_unf, input_unf.transpose(1, 2))
   
    # Fold weight gradient    
    grad_weight = grad_weight_unf.sum((0)).view(grad_weight_unf.size(1), -1, kernel_size, kernel_size)              
 
    if bias is not None and ctx.needs_input_grad[2]:
      # compute the gradients w.r.t. bias (if any)
      grad_bias = grad_output.sum((0,2,3))
  
    return grad_input, grad_weight, grad_bias, None, None

custom_conv2d = CustomConv2DFunction.apply

class CustomConv2d(Module):
  """
  The same interface as torch.nn.Conv2D
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
         padding=0, dilation=1, groups=1, bias=True):
    super(CustomConv2d, self).__init__()
    assert isinstance(kernel_size, int), "We only support squared filters"
    assert isinstance(stride, int), "We only support equal stride"
    assert isinstance(padding, int), "We only support equal padding"
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    # not used (for compatibility)
    self.dilation = dilation
    self.groups = groups

    # register weight and bias as parameters
    self.weight = nn.Parameter(torch.Tensor(
        out_channels, in_channels, kernel_size, kernel_size))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_channels))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
  	# initialization using Kaiming uniform
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in)
      nn.init.uniform_(self.bias, -bound, bound)

  def forward(self, input):
    # call our custom conv2d op
    return custom_conv2d(input, self.weight, self.bias, self.stride, self.padding)

  def extra_repr(self):
    s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
       ', stride={stride}, padding={padding}')
    if self.bias is None:
      s += ', bias=False'
    return s.format(**self.__dict__)

#################################################################################
# Part II: Design and train a network
#################################################################################
class SimpleNet(nn.Module):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(SimpleNet, self).__init__()
    # you can start from here and create a better model
    self.features = nn.Sequential(
      # conv1 block: 3x conv 3x3
      conv_op(3, 64, kernel_size=7, stride=2, padding=3),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv2 block: simple bottleneck
      conv_op(64, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv3 block: simple bottleneck
      conv_op(256, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv4 block: conv 3x3
      conv_op(256, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
    )
    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x):
    # you can implement adversarial training here
    # if self.training:
    #   # generate adversarial sample based on x
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

############################# CustomNet ############################################

class CustomNet(nn.Module):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(CustomNet, self).__init__()
    # you can start from here and create a better model
    self.conv_block1 = nn.Sequential(
      # conv1 block: 3x conv 3x3
      conv_op(3, 64, kernel_size=7, stride=2, padding=3),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
    )
    
    # max pooling 1/2
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    self.conv_relu = nn.ReLU(inplace=True)
    
    self.residual_block1 = nn.Sequential(
      # ResidualBlock
      conv_op(64, 64, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
    )
    
    self.residual_block2 = nn.Sequential(
      # ResidualBlock
      conv_op(64, 64, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
    )

    self.downsample = nn.Sequential(
      conv_op(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(256)
    )
    
    self.residual_block3 = nn.Sequential(
      # ResidualBlock
      conv_op(256, 64, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
    )
 
    self.downsample2 = nn.Sequential(
      conv_op(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(64)
    )
    
    '''self.residual_block4 = nn.Sequential(
      # ResidualBlock
      conv_op(64, 64, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
    )'''
    
    # conv4 block: conv 3x3
    self.conv_block2 = nn.Sequential(
      conv_op(256, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
    )
            
    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x):
    # you can implement adversarial training here
    # if self.training:
    #   # generate adversarial sample based on x
    # Conv Block
    x = self.conv_block1(x)    

    # Max pool
    x = self.maxpool(x)    

    # Residual Block
    residual = x
    x = self.residual_block1(x)
    x += residual
    x = self.conv_relu(x)
    
    # Residual Block
    residual = x
    x = self.residual_block2(x)    
    residual_down = self.downsample(residual)    
    x += residual_down
    x = self.conv_relu(x)

    # Max pool    
    x = self.maxpool(x)
        
    # Residual Block
    residual = x
    x = self.residual_block3(x)
    residual_down = self.downsample2(residual)    
    x += residual_down
    x = self.conv_relu(x)   
    
    # Residual Block
    residual = x
    x = self.residual_block2(x)
    residual_down = self.downsample(residual)    
    x += residual_down
    x = self.conv_relu(x)       

    # Max pool
    x = self.maxpool(x)    

    # Conv Block    
    x = self.conv_block2(x)
    
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

############################# End of CustomNet ##################################

############################# CustomNet2 ########################################
class CustomNet2(nn.Module):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(CustomNet2, self).__init__()
    self.features = nn.Sequential(
      # conv1 block: 3x conv 3x3
      conv_op(3, 64, kernel_size=7, stride=2, padding=3),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(64),


      # conv2 block: simple bottleneck
      conv_op(64, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(256),


      # conv3 block: simple bottleneck
      conv_op(256, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(256),


      # conv4 block: conv 3x3
      conv_op(256, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(512)
    )
    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)
    self.dropout = nn.Dropout(0.5)


  def forward(self, x):
    x = self.features(x)
    x = self.dropout(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
############################# End of CustomNet2 #################################

############################# ResNet ############################################

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block=ResidualBlock, layers=[2, 2, 2], num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        #self.avg_pool = nn.AvgPool2d(8)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

############################# End of ResNet ########################################

############################# GoogLeNet ############################################
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=100):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(True),
        )

        self.a3 = Inception(12, 4, 6, 8, 1, 2, 2)
        self.b3 = Inception(16, 8, 8, 12, 2, 6, 4)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(30, 12, 6, 13, 1, 3, 4)
        self.b4 = Inception(32, 10, 7, 14, 2, 4, 4)
        self.c4 = Inception(32, 8, 8, 16, 2, 4, 4)
        self.d4 = Inception(32, 7, 9, 18, 2, 4, 4)
        self.e4 = Inception(33, 16, 10, 20, 2, 8, 8)

        self.a5 = Inception(52, 16, 10, 20, 2, 8, 8)
        self.b5 = Inception(52, 24, 12, 24, 3, 8, 8)

        #self.avgpool = nn.AvgPool2d(8, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

############################# End of GoogLeNet ############################################

# change this to your model!
#default_model = SimpleNet
default_model = CustomNet2
#default_model = CustomNet
#default_model = ResNet
#default_model = GoogLeNet

#################################################################################
# Part III: Adversarial samples and Attention
#################################################################################
class PGDAttack(object):
  def __init__(self, loss_fn, num_steps=10, step_size=0.01, epsilon=0.1):
    """
    Attack a network by Project Gradient Descent. The attacker performs
    k steps of gradient descent of step size a, while always staying
    within the range of epsilon from the input image.

    Args:
      loss_fn: loss function used for the attack
      num_steps: (int) number of steps for PGD
      step_size: (float) step size of PGD
      epsilon: (float) the range of acceptable samples
               for our normalization, 0.1 ~ 6 pixel levels
    """
    self.loss_fn = loss_fn
    self.num_steps = num_steps
    self.step_size = step_size
    self.epsilon = epsilon

  def perturb(self, model, input):
    """
    Given input image X (torch tensor), return an adversarial sample
    (torch tensor) using PGD of the least confident label.

    See https://openreview.net/pdf?id=rJzIBfZAb

    Args:
      model: (nn.module) network to attack
      input: (torch tensor) input image of size N * C * H * W

    Outputs:
      output: (torch tensor) an adversarial sample of the given network
    """
    # clone the input tensor and disable the gradients
    output = input.clone()
    input.requires_grad = False

    output.requires_grad = True
    inputori = input.clone()
    net = model(inputori)
    pred = torch.min(net.data, 1)[1]
    # loop over the number of steps
    for _ in range(self.num_steps):
      #################################################################################
      # Fill in the code here
      #################################################################################
      net = model(output)
      loss = self.loss_fn(net, pred)
      loss.backward()
      temp = input + self.step_size * torch.sign(output.grad) - inputori
      temp = torch.clamp(temp, min = -self.epsilon, max=self.epsilon)
      input = temp + inputori
      output = input.clone()
      output = torch.tensor(output.data, requires_grad=True)
    return output

default_attack = PGDAttack


class GradAttention(object):
  def __init__(self, loss_fn):
    """
    Visualize a network's decision using gradients

    Args:
      loss_fn: loss function used for the attack
    """
    self.loss_fn = loss_fn

  def explain(self, model, input):
    """
    Given input image X (torch tensor), return a saliency map
    (torch tensor) by computing the max of abs values of the gradients
    given by the predicted label

    See https://arxiv.org/pdf/1312.6034.pdf

    Args:
      model: (nn.module) network to attack
      input: (torch tensor) input image of size N * C * H * W

    Outputs:
      output: (torch tensor) a saliency map of size N * 1 * H * W
    """
    # make sure input receive grads
    input.requires_grad = True
    if input.grad is not None:
      input.grad.zero_()

    #################################################################################
    # Fill in the code here
    #################################################################################
    net = model(input)
    pred = torch.max(net.data, 1)[1]
    loss = self.loss_fn(net, pred.cuda())
    loss.backward()
    middle = input.grad
    output = torch.max(torch.abs(middle), 1, True)[0]
    return output

default_attention = GradAttention

def vis_grad_attention(input, vis_alpha=2.0, n_rows=10, vis_output=None):
  """
  Given input image X (torch tensor) and a saliency map
  (torch tensor), compose the visualziations

  Args:
    input: (torch tensor) input image of size N * C * H * W
    output: (torch tensor) input map of size N * 1 * H * W

  Outputs:
    output: (torch tensor) visualizations of size 3 * HH * WW
  """
  # concat all images into a big picture
  input_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
  if vis_output is not None:
    output_maps = make_grid(vis_output.cpu(), nrow=n_rows, normalize=True)

    # somewhat awkward in PyTorch
    # add attention to R channel
    mask = torch.zeros_like(output_maps[0, :, :]) + 0.5
    mask = (output_maps[0, :, :] > vis_alpha * output_maps[0,:,:].mean())
    mask = mask.float()
    input_imgs[0,:,:] = torch.max(input_imgs[0,:,:], mask)
  output = input_imgs
  return output

default_visfunction = vis_grad_attention
