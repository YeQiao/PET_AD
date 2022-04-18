'''eEEGNet_b in PyTorch.
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
        
        # BinaryActivation(),
        # HardBinaryConv(inp, oup, 3, stride=stride, padding=(1,1,1)),
        # nn.BatchNorm3d(oup),
        # nn.PReLU(),
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
        
        # BinaryActivation(),
        # HardBinaryConv(inp, oup, 1, 1, 0),
        # nn.BatchNorm3d(oup),
        # nn.PReLU(),
    )
    
    
    #-------------------------------------------------------------------------------------------------------------------------#
class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out
        
# class BinaryActivation(nn.Module):
    # def __init__(self):
        # super(BinaryActivation, self).__init__()

    # def forward(self, x):
        # out_forward = torch.sign(x)
        # #out_e1 = (x^2 + 2*x)
        # out_e2 = (-x^2 + 2*x)
        # out_e_total = 0
        # mask1 = x < -0.2
        # mask2 = x < 0
        # mask3 = x < 0.2
        # out1 = (-1) * mask1.type(torch.float32) + (25*x*x + 10*x) * (1-mask1.type(torch.float32))
        # out2 = out1 * mask2.type(torch.float32) + ((-25)*x*x + 10*x) * (1-mask2.type(torch.float32))
        # out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        # out = out_forward.detach() - out3.detach() + out3

        # return out

# class BinaryActivation(nn.Module):
    # def __init__(self):
        # super(BinaryActivation, self).__init__()

    # def forward(self, x):
        # out_forward = torch.sign(x)
        # out_e_total = 0
        # mask1 = x < -1
        # mask2 = (-1 < x) & (x < 0)
        # mask3 = (0 < x) & (x < 1)
        # mask4 = x > 1

        # out1 = (-1) * mask1.type(torch.float32)
        # out2 = ((-1)*torch.sqrt(1-(x-1)**2)) * mask2.type(torch.float32)
        # out3 = (torch.sqrt(1-(x-1)**2)) * mask3.type(torch.float32)
        # out4 = 1 * mask4.type(torch.float32)
        # out = out1 + out2 + out3 + out4
        
        # return out

# class HardBinaryConv(nn.Conv3d):
    # def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups = None):
        # super(HardBinaryConv, self).__init__(in_chn, out_chn, kernel_size, stride, padding, groups)
        # self.stride = stride
        # self.padding = padding
        # self.kernel_size = kernel_size
        # self.groups = groups
        # self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size * kernel_size
        # self.shape = (out_chn, in_chn, kernel_size, kernel_size, kernel_size)
        # self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    # def forward(self, x):
        # real_weights = self.weights.view(self.shape)
        # scaling_factor = torch.mean(torch.mean(torch.mean(torch.mean(abs(real_weights),dim=4,keepdim=True),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        
        
        # print('scaling_factor',scaling_factor, flush=True)
        # scaling_factor = scaling_factor.detach()
        # binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        # cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        # binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        
        # print(binary_weights, flush=True)
        # y = nn.Conv3d(x, binary_weights, self.kernel_size, stride=self.stride, padding=self.padding)

        # return y
        
        
        
class HardBinaryConv(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        super(HardBinaryConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias)

        self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size, kernel_size)) * 0.001, requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        w = self.weight
        # alpha = torch.mean(
            # torch.mean(torch.mean(abs(w), dim=4, keepdim=True), dim=3, keepdim=True), dim=2, keepdim=True).detach()
        alpha = torch.mean(torch.mean(
            torch.mean(torch.mean(abs(w), dim=4, keepdim=True), dim=3, keepdim=True), dim=2, keepdim=True),dim=1,keepdim=True).detach()
        w_alpha = alpha * torch.sign(w)
        # bx = BinaryActivation()(x)
        cliped_bw = torch.clamp(w, -1.0, 1.0)
        bw = w_alpha.detach() - cliped_bw.detach() + cliped_bw  # let the gradient of binary function to 1.

        output = F.conv3d(x, bw, self.bias, self.stride, self.padding)

        return output
    #-------------------------------------------------------------------------------------------------------------------------#
    
    
# class InvertedResidual(nn.Module)
    # def __init__(self, inp, oup, stride, expand_ratio):
        # super(InvertedResidual, self).__init__()
        # self.stride = stride

        # hidden_dim = round(inp * expand_ratio)
        # self.use_res_connect = self.stride == (1,1,1) and inp == oup

        # if expand_ratio == 1:
            # self.conv = nn.Sequential(
                ##dw
                   # HardBinaryConv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # nn.BatchNorm3d(hidden_dim),
                   # BinaryActivation(),
                ##pw-linear
                   # HardBinaryConv(hidden_dim, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm3d(oup),
            # )
        # else:
            # self.conv = nn.Sequential(
                ##pw
                   # HardBinaryConv(hidden_dim, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm3d(hidden_dim),
                   # BinaryActivation(),
                ##dw
                   # HardBinaryConv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # nn.BatchNorm3d(hidden_dim),
                   # BinaryActivation(),
                ##pw-linear
                   # HardBinaryConv(hidden_dim, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm3d(oup),
            # )

    # def forward(self, x):
        # if self.use_res_connect:
            # return x + self.conv(x)
        # else:
            # return self.conv(x)
            
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1,1,1)  and inp == oup
       
        
        self.downsampling = nn.AvgPool3d(kernel_size=2, stride=stride)

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                # nn.BatchNorm3d(hidden_dim),
                BinaryActivation(),
                HardBinaryConv(hidden_dim, hidden_dim, 3, stride, 1),
                nn.BatchNorm3d(hidden_dim),
                # nn.PReLU(),
               
                # pw-linear
                # nn.BatchNorm3d(hidden_dim),
                BinaryActivation(),
                HardBinaryConv(hidden_dim, oup, 1, 1, 0),
                nn.BatchNorm3d(oup),
                
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # nn.BatchNorm3d(inp),
                BinaryActivation(),
                HardBinaryConv(inp, hidden_dim, 1, 1, 0),
                nn.BatchNorm3d(hidden_dim),
                # nn.PReLU(),
                # dw
                # nn.BatchNorm3d(hidden_dim),
                BinaryActivation(),
                HardBinaryConv(hidden_dim, hidden_dim, 3, stride, 1),
                nn.BatchNorm3d(hidden_dim),
                # nn.PReLU(),
                # pw-linear
                # nn.BatchNorm3d(hidden_dim),
                BinaryActivation(),
                HardBinaryConv(hidden_dim, oup, 1, 1, 0),
                nn.BatchNorm3d(oup),

            )

    def forward(self, x):
        residual = x
        if self.stride != (1,1,1):
            residual = self.downsampling(x)
        return residual + self.conv(x)
        
        
    # def forward(self, x):
        # if self.use_res_connect:
            # return x + self.conv(x)
        # else:
            # return self.conv(x)
            
            
class eEEGNet_b(nn.Module):
    def __init__(self, num_classes=1000, sample_size=224, width_mult=1., model_name = 'V1'):
        super(eEEGNet_b, self).__init__()
        block = InvertedResidual
        input_channel = 16

        # V1
        if model_name == 'V1':
            last_channel = 320
            expension_t = 2
            width_mult = 0.4
            
        # V2
        if model_name == 'V2':
            last_channel = 640
            expension_t = 3
            width_mult = 0.5
        
        # V3
        if model_name == 'V3':
            last_channel = 640
            expension_t = 4
            width_mult = 0.8

        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, (1,1,1)],
            [expension_t, 16, 2, (2,2,2)],
            # [6,  32, 3, (2,2,2)],
            # [6,  64, 4, (2,2,2)],
            # [6,  96, 3, (1,1,1)],
            # [6, 160, 3, (2,2,2)],
            # [6, 320, 1, (1,1,1)], 
        ]

        # building first layer
        assert sample_size % 16 == 0.
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, (1,2,2))]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1,1,1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        # self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        self.last_layer = conv_1x1x1_bn(input_channel, self.last_channel)
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = self.last_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_fine_tuning_parameters_b(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")

    
def get_model_b(**kwargs):
    """
    Returns the model.
    """
    model = eEEGNet_b(**kwargs)
    return model


if __name__ == "__main__":
    model = get_model_b(num_classes=600, sample_size=112, width_mult=1., model_name = 'V1')
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)


    input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    output = model(input_var)
    print(output.shape)