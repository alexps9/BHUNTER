import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import cv2
import random
from models import vgg

class UnsupModel(BaseModel):
  def name(self):
    return 'UnsupModel'

  def initialize(self, opt):
    BaseModel.initialize(self, opt)

    nb = opt.batchSize
    size = opt.fineSize
    self.input_A = self.Tensor(nb, opt.input_nc, size, size)
    self.input_B = self.Tensor(nb, opt.output_nc, size, size)

    self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                    opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
                                    self.gpu_ids)
    self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                    opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
                                    self.gpu_ids)

    if self.isTrain and (self.opt.lambda_content_A > 0 or self.opt.lambda_content_B > 0):
      self.vgg = vgg.vgg.cuda(self.gpu_ids[0])
      vgg_pretrained_path = 'saved_models/vgg_normalised.pth'
      self.vgg.load_state_dict(torch.load(vgg_pretrained_path))
      self.vgg.eval()
      self.vgg.requires_grad = False
      self.instance_norm_layer = torch.nn.InstanceNorm2d(512, affine=False)

    if self.isTrain:
      use_sigmoid = opt.no_lsgan
      self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                      opt.which_model_netD,
                                      opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
      self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                      opt.which_model_netD,
                                      opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
    if not self.isTrain or opt.continue_train:
      which_epoch = opt.which_epoch
      self.load_network(self.netG_A, 'G_A', which_epoch)
      self.load_network(self.netG_B, 'G_B', which_epoch)
      if self.isTrain:
        self.load_network(self.netD_A, 'D_A', which_epoch)
        self.load_network(self.netD_B, 'D_B', which_epoch)

    if self.isTrain:
      self.old_lr = opt.lr
      self.fake_A_pool = ImagePool(opt.pool_size)
      self.fake_B_pool = ImagePool(opt.pool_size)
      # define loss functions
      self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
      self.criterionCycle = torch.nn.L1Loss()
      self.criterionIdt = torch.nn.L1Loss()
      self.criterionUnsup = torch.nn.L1Loss()
      self.criterionCont = torch.nn.MSELoss()
      # initialize optimizers
      self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), 
                                          lr=opt.lr, betas=(opt.beta1, 0.999))
      self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
      self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
      self.optimizers = []
      self.schedulers = []
      self.optimizers.append(self.optimizer_G)
      self.optimizers.append(self.optimizer_D_A)
      self.optimizers.append(self.optimizer_D_B)
      for optimizer in self.optimizers:
        self.schedulers.append(networks.get_scheduler(optimizer, opt))

    print('---------- Networks initialized -------------')
    networks.print_network(self.netG_A)
    networks.print_network(self.netG_B)
    if self.isTrain:
      networks.print_network(self.netD_A)
      networks.print_network(self.netD_B)
    print('-----------------------------------------------')

  def GaussianNoise(self, ins, mean=0, stddev=0.03):
    # adapted from https://github.com/daooshee/ReReVST-Code/blob/master/train/loss_networks.py
    stddev = stddev + random.random() * stddev
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    
    if ins.is_cuda:
        noise = noise.cuda()
    return ins + noise

  def GenerateFakeFlow(self, height, width, motion_level, shift_level, scale_level):
    # adapted from https://github.com/daooshee/ReReVST-Code/blob/master/train/loss_networks.py
    ''' height: img.shape[0]
        width:  img.shape[1] '''
    if scale_level > 0:
        flow = np.ones([height,width,2])
        scale_factor = random.uniform(-scale_level, scale_level)
        unit = np.arange(-3, 3, 0.01)
        cent_x = random.randint(0, len(unit) - width)
        cent_y = random.randint(0, len(unit) - height)

        x_range = unit[cent_x:cent_x+width]
        y_range = unit[cent_y:cent_y+height]

        xx = np.tile(x_range.reshape(1,-1), (height,1)) * scale_factor
        yy = np.tile(y_range.reshape(-1,1), (1,width)) * scale_factor

        flow[:,:,0] = xx
        flow[:,:,1] = yy

        return torch.from_numpy(flow.transpose((2, 0, 1))).float()

    if motion_level > 0:
        flow = np.random.normal(0, scale=motion_level, size = [height//100, width//100, 2])
        flow = cv2.resize(flow, (width, height))
        flow[:,:,0] += random.randint(-shift_level, shift_level)
        flow[:,:,1] += random.randint(-shift_level, shift_level)
        flow = cv2.blur(flow,(100,100))
    else:
        flow = np.ones([height,width,2])
        flow[:,:,0] = random.randint(-shift_level, shift_level)
        flow[:,:,1] = random.randint(-shift_level, shift_level)

    return torch.from_numpy(flow.transpose((2, 0, 1))).float()

  def GenerateFakeData(self, first_frame, hyperparameters):
    # adapted from https://github.com/daooshee/ReReVST-Code/blob/master/train/loss_networks.py
    B, C, H, W = first_frame.size()

    fake_flow = self.GenerateFakeFlow(H, W, hyperparameters.motion_level, hyperparameters.shift_level, hyperparameters.scale_level)
    if first_frame.is_cuda:
        fake_flow = fake_flow.cuda()
    fake_flow = fake_flow.expand(B, 2, H, W)
    second_frame = util.warp(first_frame, fake_flow)
    second_frame = self.GaussianNoise(second_frame, stddev=hyperparameters.noise_level)

    return second_frame, fake_flow

  def set_input(self, input):
    AtoB = self.opt.which_direction == 'AtoB'
    input_A = input['A1']
    input_B = input['B1']

    self.input_A.resize_(input_A.size()).copy_(input_A)
    self.input_B.resize_(input_B.size()).copy_(input_B)

    self.image_paths = input['A_paths' if AtoB else 'B_paths']

  def forward(self):
    self.real_A = Variable(self.input_A)
    self.real_B = Variable(self.input_B)

  def test(self):
    with torch.no_grad():

      self.real_A = Variable(self.input_A)
      self.real_B = Variable(self.input_B)

      self.fake_B = self.netG_A(self.real_A)
      self.fake_A = self.netG_B(self.real_B)

  def get_image_paths(self):
    return self.image_paths

  def backward_D_basic(self, netD, real, fake):
    # Real
    pred_real = netD(real)
    loss_D_real = self.criterionGAN(pred_real, True)
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake = self.criterionGAN(pred_fake, False)
    # Combined loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5 
    # backward
    loss_D.backward()
    return loss_D, pred_real, pred_fake

  def backward_D_A(self):
    fake_B = self.fake_B_pool.query(self.fake_B)
    loss_D_A, pred_real_B, pred_fake_B = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    self.loss_D_A = loss_D_A.data
    self.pred_real_B = pred_real_B.data
    self.pred_fake_B = pred_fake_B.data

  def backward_D_B(self):
    fake_A = self.fake_A_pool.query(self.fake_A)
    loss_D_B, pred_real_A, pred_fake_A = self.backward_D_basic(self.netD_B, self.real_A, fake_A)


    self.loss_D_B = loss_D_B.data
    self.pred_real_A = pred_real_A.data
    self.pred_fake_A = pred_fake_A.data

  def backward_G(self):
    # GAN loss D_A(G_A(A))
    self.fake_B = self.netG_A(self.real_A)
    pred_fake = self.netD_A(self.fake_B)
    loss_G_A = self.criterionGAN(pred_fake, True)

    self.fake_A = self.netG_B(self.real_B)
    pred_fake = self.netD_B(self.fake_A)
    loss_G_B = self.criterionGAN(pred_fake, True) 

    # Forward cycle loss
    self.rec_A = self.netG_B(self.fake_B)
    loss_cycle_A = self.opt.lambda_cycle_A * self.criterionCycle(self.rec_A, self.real_A)

    # Backward cycle loss
    self.rec_B = self.netG_A(self.fake_A)
    loss_cycle_B = self.opt.lambda_cycle_B *self.criterionCycle(self.rec_B, self.real_B)

    # Unsup Forward cycle loss
    if self.opt.lambda_unsup_cycle_A > 0:
      warped_fake_B, fake_flow_fake_B = self.GenerateFakeData(self.fake_B, self.opt)
      recon_warped_real_A = self.netG_B(warped_fake_B)
      real_A_warped = util.warp(self.real_A, fake_flow_fake_B)
      diff_cycle_A = torch.abs(recon_warped_real_A - real_A_warped) 
      loss_unsup_cycle_A = self.opt.lambda_unsup_cycle_A * torch.mean(diff_cycle_A)
    else:
      loss_unsup_cycle_A = torch.tensor(0.)

    # Unsup Backward cycle loss
    if self.opt.lambda_unsup_cycle_B > 0:
      warped_fake_A, fake_flow_fake_A = self.GenerateFakeData(self.fake_A, self.opt)
      recon_warped_real_B = self.netG_A(warped_fake_A)
      real_B_warped = util.warp(self.real_B, fake_flow_fake_A)
      diff_cycle_B = torch.abs(recon_warped_real_B - real_B_warped) 
      loss_unsup_cycle_B = self.opt.lambda_unsup_cycle_B * torch.mean(diff_cycle_B)
    else:
      loss_unsup_cycle_B = torch.tensor(0.)

    # unsupervised_loss
    if self.opt.lambda_spa_unsup_A > 0:
      warped_real_A, fake_flow_A = self.GenerateFakeData(self.real_A, self.opt)
      warped_fake_B = self.netG_A(warped_real_A)
      fake_B_warped = util.warp(self.fake_B, fake_flow_A)
      diff_A = torch.abs(warped_fake_B - fake_B_warped) 
      unsup_loss_A = self.opt.lambda_spa_unsup_A * torch.mean(diff_A)
    else:
      unsup_loss_A = torch.tensor(0.)
      self.diff_A_ratio = 0

    if self.opt.lambda_spa_unsup_B > 0:
      warped_real_B, fake_flow_B = self.GenerateFakeData(self.real_B, self.opt)
      warped_fake_A = self.netG_B(warped_real_B)
      fake_A_warped = util.warp(self.fake_A, fake_flow_B)
      diff_B = torch.abs(warped_fake_A - fake_A_warped) 
      unsup_loss_B = self.opt.lambda_spa_unsup_B * torch.mean(diff_B)
    else:
      unsup_loss_B = torch.tensor(0.)
      self.diff_B_ratio = 0

    # perceptual loss
    if self.opt.lambda_content_A > 0 or self.opt.lambda_content_B > 0:
      in_feat_fake_B = self.instance_norm_layer(self.vgg(self.fake_B))
      in_feat_real_A = self.instance_norm_layer(self.vgg(self.real_A))
      in_feat_fake_A = self.instance_norm_layer(self.vgg(self.fake_A))
      in_feat_real_B = self.instance_norm_layer(self.vgg(self.real_B))

      self.cont_diff_A = torch.sum(torch.abs(in_feat_fake_B - in_feat_real_A), dim=1, keepdim=True)
      self.cont_diff_A = self.cont_diff_A / torch.max(self.cont_diff_A) * 2 - 1
      self.cont_diff_B = torch.sum(torch.abs(in_feat_fake_A - in_feat_real_B), dim=1, keepdim=True)
      self.cont_diff_B = self.cont_diff_B / torch.max(self.cont_diff_B) * 2 - 1

      cont_loss_A = self.opt.lambda_content_A * self.criterionCont(in_feat_fake_B, in_feat_real_A)
      cont_loss_B = self.opt.lambda_content_B * self.criterionCont(in_feat_fake_A, in_feat_real_B)
    else:
      cont_loss_A, cont_loss_B = torch.tensor(0.), torch.tensor(0.)

    if self.opt.lambda_identity > 0:
      # G_A should be identity if real_B is fed.
      self.idt_A = self.netG_A(self.real_B)
      loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_identity
      # G_B should be identity if real_A is fed.
      self.idt_B = self.netG_B(self.real_A)
      loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.opt.lambda_identity

    else:
      loss_idt_A, loss_idt_B = torch.tensor(0.), torch.tensor(0.)

    # combined loss
    loss_G = (loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_unsup_cycle_A + loss_unsup_cycle_B +\
               unsup_loss_A + unsup_loss_B + cont_loss_A + cont_loss_B + loss_idt_A + loss_idt_B)
    loss_G.backward()

    self.loss_G_A = loss_G_A.data
    self.loss_G_B = loss_G_B.data
    self.loss_cycle_A = loss_cycle_A.data
    self.loss_cycle_B = loss_cycle_B.data
    self.loss_unsup_cycle_A = loss_unsup_cycle_A.data
    self.loss_unsup_cycle_B = loss_unsup_cycle_B.data
    self.unsup_loss_A = unsup_loss_A.data
    self.unsup_loss_B = unsup_loss_B.data
    self.cont_loss_A = cont_loss_A.data
    self.cont_loss_B = cont_loss_B.data
    self.loss_idt_A = loss_idt_A.data
    self.loss_idt_B = loss_idt_B.data


  def optimize_parameters(self):
    # forward
    self.forward()
    # G_A and G_B
    self.optimizer_G.zero_grad()
    self.backward_G()
    self.optimizer_G.step()
    # D_A
    self.optimizer_D_A.zero_grad()
    self.backward_D_A()
    self.optimizer_D_A.step()
    # D_B
    self.optimizer_D_B.zero_grad()
    self.backward_D_B()
    self.optimizer_D_B.step()

  def get_current_errors(self):
    ret_errors = OrderedDict(
      [('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A), ('UnCyc_A', self.loss_unsup_cycle_A), 
      ('Unsup_A', self.unsup_loss_A), ('Cont_A', self.cont_loss_A), ('Idt_A', self.loss_idt_A),
      ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B', self.loss_cycle_B), ('UnCyc_B', self.loss_unsup_cycle_B), 
      ('Unsup_B', self.unsup_loss_B), ('Cont_B', self.cont_loss_B), ('Idt_B', self.loss_idt_B),
      ])

    return ret_errors

  def get_current_visuals(self):
    real_A = util.tensor2im(self.input_A)
    fake_B = util.tensor2im(self.fake_B.data)
    real_B = util.tensor2im(self.input_B.data)
    fake_A = util.tensor2im(self.fake_A.data)

    ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B),
                               ('real_B', real_B), ('fake_A', fake_A),
                               ])

    return ret_visuals

  def save(self, label):
    self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
    self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
    self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
    self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
