
import jittor as jt
from jittor import init
from jittor import nn
from jittor import lr_scheduler
import functools
from jittor import optim
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        init.gauss_(m.weight.data, mean=0.0, std=0.02)
    elif (classname.find('Linear') != (- 1)):
        init.gauss_(m.weight.data, mean=0.0, std=0.02)
    elif (classname.find('BatchNorm2d') != (- 1)):
        init.gauss_(m.weight.data, mean=1.0, std=0.02)
        init.constant_(m.bias.data, value=0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        init.xavier_gauss_(m.weight.data,gain=0.02)
    elif (classname.find('Linear') != (- 1)):
        init.xavier_gauss_(m.weight.data,gain=0.02)
    elif (classname.find('BatchNorm2d') != (- 1)):
        init.gauss_(m.weight.data, mean=1.0, std=0.02)
        init.constant_(m.bias.data, value=0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif (classname.find('Linear') != (- 1)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif (classname.find('BatchNorm2d') != (- 1)):
        init.gauss_(m.weight.data, mean=1.0, std=0.02)
        init.constant_(m.bias.data, value=0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        ##TODO:正交初始化
        init.orthogonal_(m.weight.data, gain=1)
    elif (classname.find('Linear') != (- 1)):
        init.orthogonal_(m.weight.data, gain=1)
    elif (classname.find('BatchNorm2d') != (- 1)):
        init.gauss_(m.weight.data, mean=1.0, std=0.02)
        init.constant_(m.bias.data, value=0.0)

def init_weights(net, init_type='normal'):
    print(('initialization method [%s]' % init_type))
    if (init_type == 'normal'):
        net.apply(weights_init_normal)
    elif (init_type == 'xavier'):
        net.apply(weights_init_xavier)
    elif (init_type == 'kaiming'):
        net.apply(weights_init_kaiming)
    elif (init_type == 'orthogonal'):
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(('initialization method [%s] is not implemented' % init_type))

def get_norm_layer(norm_type='instance'):
    if (norm_type == 'batch'):
        norm_layer = functools.partial(nn.BatchNorm, affine=True)
    elif (norm_type == 'instance'):
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif (norm_type == 'none'):
        norm_layer = None
    else:
        raise NotImplementedError(('normalization layer [%s] is not found' % norm_type))
    return norm_layer

def get_scheduler(optimizer, opt):
    if (opt.lr_policy == 'lambda'):

        def lambda_rule(epoch):
            lr_l = (1.0 - (max(0, (((epoch + 1) + opt.epoch_count) - opt.niter)) / float((opt.niter_decay + 1))))
            return lr_l
        scheduler = optim.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif (opt.lr_policy == 'step'):
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif (opt.lr_policy == 'plateau'):
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal'):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)
    if (which_model_netG == 'resnet_9blocks'):
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif (which_model_netG == 'resnet_6blocks'):
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif (which_model_netG == 'unet_128'):
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif (which_model_netG == 'unet_256'):
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError(('Generator model name [%s] is not recognized' % which_model_netG))
    init_weights(netG, init_type=init_type)
    return netG

def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal'):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)
    if (which_model_netD == 'basic'):
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif (which_model_netD == 'n_layers'):
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif (which_model_netD == 'pixel'):
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif (which_model_netD == 'global'):
        netD = GlobalDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif (which_model_netD == 'global_np'):
        netD = GlobalNPDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError(('Discriminator model name [%s] is not recognized' % which_model_netD))
    init_weights(netD, init_type=init_type)
    return netD

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print(('Total number of parameters: %d' % num_params))

class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var=jt.full(input.shape,self.real_label).stop_grad()
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var=jt.full(input.shape,self.fake_label).stop_grad()
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if (type(norm_layer) == functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)
        model = [nn.ReflectionPad2d(3), nn.Conv(input_nc, ngf, 7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU()]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = (2 ** i)
            model += [nn.Conv((ngf * mult), ((ngf * mult) * 2), 3, stride=2, padding=1, bias=use_bias), norm_layer(((ngf * mult) * 2)), nn.ReLU()]
        mult = (2 ** n_downsampling)
        for i in range(n_blocks):
            model += [ResnetBlock((ngf * mult), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        for i in range(n_downsampling):
            mult = (2 ** (n_downsampling - i))
            model += [nn.ConvTranspose((ngf * mult), int(((ngf * mult) / 2)), 3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(((ngf * mult) / 2))), nn.ReLU()]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv(ngf, output_nc, 7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def execute(self, input):
        return self.model(input)

class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if (padding_type == 'reflect'):
            conv_block += [nn.ReflectionPad2d(1)]
        elif (padding_type == 'replicate'):
            conv_block += [nn.ReplicationPad2d(1)]
        elif (padding_type == 'zero'):
            p = 1
        else:
            raise NotImplementedError(('padding [%s] is not implemented' % padding_type))
        conv_block += [nn.Conv(dim, dim, 3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        else:
            conv_block += [nn.Dropout(0)]
        p = 0
        if (padding_type == 'reflect'):
            conv_block += [nn.ReflectionPad2d(1)]
        elif (padding_type == 'replicate'):
            conv_block += [nn.ReplicationPad2d(1)]
        elif (padding_type == 'zero'):
            p = 1
        else:
            raise NotImplementedError(('padding [%s] is not implemented' % padding_type))
        conv_block += [nn.Conv(dim, dim, 3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def execute(self, x):
        out = (x + self.conv_block(x))
        return out

class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock((ngf * 8), (ngf * 8), input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range((num_downs - 5)):
            unet_block = UnetSkipConnectionBlock((ngf * 8), (ngf * 8), input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock((ngf * 4), (ngf * 8), input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock((ngf * 2), (ngf * 4), input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, (ngf * 2), input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block

    def execute(self, input):
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if (type(norm_layer) == functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)
        if (input_nc is None):
            input_nc = outer_nc
        downconv = nn.Conv(input_nc, inner_nc, 4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(scale=0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose((inner_nc * 2), outer_nc, 4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = ((down + [submodule]) + up)
        elif innermost:
            upconv = nn.ConvTranspose(inner_nc, outer_nc, 4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = (down + up)
        else:
            upconv = nn.ConvTranspose((inner_nc * 2), outer_nc, 4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = (((down + [submodule]) + up) + [nn.Dropout(0.5)])
            else:
                model = ((down + [submodule]) + up)
        self.model = nn.Sequential(*model)

    def execute(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return jt.contrib.concat([x, self.model(x)], dim=1)

class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if (type(norm_layer) == functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)
        kw = 4
        padw = 1
        sequence = [nn.Conv(input_nc, ndf, kw, stride=2, padding=padw), nn.LeakyReLU(scale=0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min((2 ** n), 8)
            sequence += [nn.Conv((ndf * nf_mult_prev), (ndf * nf_mult), kw, stride=2, padding=padw, bias=use_bias), norm_layer((ndf * nf_mult)), nn.LeakyReLU(scale=0.2)]
        nf_mult_prev = nf_mult
        nf_mult = min((2 ** n_layers), 8)
        sequence += [nn.Conv((ndf * nf_mult_prev), (ndf * nf_mult), kw, stride=1, padding=padw, bias=use_bias), norm_layer((ndf * nf_mult)), nn.LeakyReLU(scale=0.2)]
        sequence += [nn.Conv((ndf * nf_mult), 1, kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def execute(self, input):
        return self.model(input)

class GlobalDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm, use_sigmoid=False):
        super(GlobalDiscriminator, self).__init__()
        if (type(norm_layer) == functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)
        kw = 4
        padw = 1
        sequence = [nn.Conv(input_nc, ndf, kw, stride=2, padding=padw), nn.LeakyReLU(scale=0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min((2 ** n), 8)
            sequence += [nn.Conv((ndf * nf_mult_prev), (ndf * nf_mult), kw, stride=2, padding=padw, bias=use_bias), norm_layer((ndf * nf_mult)), nn.LeakyReLU(scale=0.2)]
        nf_mult_prev = nf_mult
        nf_mult = min((2 ** n_layers), 8)
        sequence += [nn.Conv((ndf * nf_mult_prev), (ndf * nf_mult), kw, stride=2, padding=padw, bias=use_bias), norm_layer((ndf * nf_mult)), nn.LeakyReLU(scale=0.2)]
        sequence += [nn.Conv((ndf * nf_mult), 1, kw, stride=2, padding=0)]
        sequence += [nn.Conv(1, 1, 7, stride=1, padding=0)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def execute(self, input):
        return self.model(input)

class GlobalNPDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm, use_sigmoid=False):
        super(GlobalNPDiscriminator, self).__init__()
        if (type(norm_layer) == functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)
        kw = [8, 3, 4]
        padw = 0
        sequence = [nn.Conv(input_nc, ndf, kw[0], stride=2, padding=padw), nn.LeakyReLU(scale=0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min((2 ** n), 8)
            sequence += [nn.Conv((ndf * nf_mult_prev), (ndf * nf_mult), kw[n], stride=2, padding=padw, bias=use_bias), norm_layer((ndf * nf_mult)), nn.LeakyReLU(scale=0.2)]
        nf_mult_prev = nf_mult
        nf_mult = min((2 ** n_layers), 8)
        sequence += [nn.Conv((ndf * nf_mult_prev), (ndf * nf_mult), 4, stride=2, padding=padw, bias=use_bias), norm_layer((ndf * nf_mult)), nn.LeakyReLU(scale=0.2)]
        sequence += [nn.Conv((ndf * nf_mult), 1, 4, stride=2, padding=0)]
        sequence += [nn.Conv(1, 1, 6, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def execute(self, input):
        return self.model(input)

class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if (type(norm_layer) == functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)
        self.net = [nn.Conv(input_nc, ndf, 1, stride=1, padding=0), nn.LeakyReLU(scale=0.2), nn.Conv(ndf, (ndf * 2), 1, stride=1, padding=0, bias=use_bias), norm_layer((ndf * 2)), nn.LeakyReLU(scale=0.2), nn.Conv((ndf * 2), 1, 1, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def execute(self, input):
        return self.net(input)
