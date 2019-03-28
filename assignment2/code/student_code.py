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

class CustomNet(nn.Module):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(SimpleNet, self).__init__()
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
      conv_op(64, 256, kernel_size=3, stride=stride, padding=1, bias=False),
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
      conv_op(256, 64, kernel_size=3, stride=stride, padding=1, bias=False),
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

    # Conv Block    
    x = self.conv_block2(x)
    
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

# change this to your model!
default_model = CustomNet

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

    # loop over the number of steps
    for _ in range(self.num_steps):
      #################################################################################
      # Fill in the code here
      #################################################################################
      net = model(output)
      pred = torch.min(net.data, 1)[1]
      print(net.size())
      label = torch.FloatTensor(net.size()).fill_(0)
      label[pred] = 1
      loss = self.loss_fn(net, label)
      loss.backward()
      inputgrad = input.grad
      output = output + self.epsilon * torch.sign(inputgrad)
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
