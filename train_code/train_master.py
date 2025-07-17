# -*- coding: utf-8 -*-

import os, sys

import scipy
import torch
import glob
import time, shutil
import math
import gc
from tqdm import tqdm
from collections import defaultdict

# torch module import
from torch.multiprocessing import Pool, Process, set_start_method
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

try:
    set_start_method('spawn')
except RuntimeError:
    pass

# import files from local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from loss.gan_loss import GANLoss, MultiScaleGANLoss
from loss.pixel_loss import PixelLoss, L1_Charbonnier_loss
from loss.perceptual_loss import PerceptualLoss
from loss.anime_perceptual_loss import Anime_PerceptualLoss
from architecture.dataset import ImageDataset
from scripts.generate_lr_esr import generate_low_res_esr
from torchvision.utils import save_image
import numpy as np
import pywt

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

def wavelet_similarity(signal1, signal2, wavelet='haar', mode='reflect', level=1):
    # 将信号进行小波分解
    coeffs1 = pywt.wavedec(signal1, wavelet, level=level, mode=mode)
    coeffs2 = pywt.wavedec(signal2, wavelet, level=level, mode=mode)
    # print(f"{signal1}<+++++>{signal2}")
    # print(f"{coeffs1}====>{coeffs2}")

    # 计算每个子带的能量
    similarity = 0
    for c1, c2 in zip(coeffs1, coeffs2):
        similarity += np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))

    # print(f"===>{similarity}")
    return similarity / len(coeffs1)


def reg(x, y):
    cova_x = (x.t() @ x)
    cova_y = (y.t() @ y)

    mean_diff = (x - y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()

    return mean_diff + cova_diff


def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())

    mid = torch.matmul(x.cuda(), w.t().cuda())

    mid = mid + b.cuda()
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0].cuda()
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())
    else:
        Z = Z * torch.cat((torch.cos(mid).cuda(), torch.sin(mid).cuda()), dim=-1)

    return Z


class train_master(object):
    def __init__(self, options, args, model_name, has_discriminator=False) -> None:
        # General specs setup
        self.args = args
        self.model_name = model_name
        self.options = options
        self.has_discriminator = has_discriminator

        # Loss init
        self.loss_init()

        # Generator
        self.call_model()  # generator + discriminator...

        # Optimizer
        self.learning_rate = options['start_learning_rate']
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate,
                                            betas=(options["adam_beta1"], options["adam_beta2"]))
        if self.has_discriminator:
            self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate,
                                                betas=(self.options["adam_beta1"], self.options["adam_beta2"]))

        # Train specs
        self.start_iteration = 0
        self.lowest_generator_loss = float("inf")

        # Other auxiliary function
        self.writer = SummaryWriter()
        self.weight_store = defaultdict(int)

        # Options setting
        self.n_iterations = options['train_iterations']
        self.batch_size = options['train_batch_size']
        self.n_cpu = options['train_dataloader_workers']

    def adjust_learning_rate(self, iteration_idx):
        self.learning_rate = self.options['start_learning_rate']
        end_iteration = self.options['train_iterations']

        # Calculate a learning rate we need in real-time based on the iteration_idx
        for idx in range(min(end_iteration, iteration_idx) // self.options['decay_iteration']):
            idx = idx + 1
            if idx * self.options['decay_iteration'] in self.options['double_milestones']:
                # double the learning rate in milestones
                self.learning_rate = self.learning_rate * 2
            else:
                # else, try to multiply decay_gamma (when we decay, we won't upscale)
                self.learning_rate = self.learning_rate * self.options[
                    'decay_gamma']  # should be divisible in all cases

        # Change the learning rate to our target
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = self.learning_rate

        if self.has_discriminator:
            # print("We didn't yet handle discriminator, but we think that it should be necessary")
            for param_group in self.optimizer_d.param_groups:
                param_group['lr'] = self.learning_rate

        assert (self.learning_rate == self.optimizer_g.param_groups[0]['lr'])

    def pixel_loss_load(self):
        if self.options['pixel_loss'] == "L1":
            self.cri_pix = PixelLoss().cuda()
        elif self.options['pixel_loss'] == "L1_Charbonnier":
            self.cri_pix = L1_Charbonnier_loss().cuda()

        print("We are using {} loss".format(self.options['pixel_loss']))

    def GAN_loss_load(self):
        # parameter init
        gan_loss_weight = self.options["gan_loss_weight"]
        vgg_type = self.options['train_perceptual_vgg_type']

        # Preceptual Loss
        self.cri_pix = torch.nn.L1Loss().cuda()
        self.cri_vgg_perceptual = PerceptualLoss(self.options['vgg_perceptual_layer_weights'], vgg_type,
                                                 perceptual_weight=self.options["vgg_perceptual_loss_weight"]).cuda()
        self.cri_danbooru_perceptual = Anime_PerceptualLoss(self.options["danbooru_layer_weights"],
                                                            perceptual_weight=self.options[
                                                                "danbooru_perceptual_loss_weight"]).cuda()

        # GAN loss
        if self.options['discriminator_type'] == "PatchDiscriminator":
            self.cri_gan = MultiScaleGANLoss(gan_type="lsgan",
                                             loss_weight=gan_loss_weight).cuda()  # already put in loss scaler for discriminator
        elif self.options['discriminator_type'] == "UNetDiscriminator":
            self.cri_gan = GANLoss(gan_type="vanilla",
                                   loss_weight=gan_loss_weight).cuda()  # already put in loss scaler for discriminator

    def tensorboard_epoch_draw(self, epoch_loss, epoch):
        self.writer.add_scalar('Loss/train-Loss-Epoch', epoch_loss, epoch)

    def master_run(self):
        torch.backends.cudnn.benchmark = True
        print("options are ", self.options)

        # Generate a new LR dataset before doing anything (Must before Data Loading)  ---> 两次配对Simple-Align
        # pass
        # t1 = 1
        # self.generate_lr(t1)
        # self.generate_lr(t1 + 1)

        # Load data
        train_lr_paths = glob.glob(self.options["lr_dataset_path"] + "/*.*")
        train_lr_reg_paths = glob.glob(self.options["lr_dataset_path_reg"] + "/*.*")
        degrade_hr_paths = glob.glob(self.options["degrade_hr_dataset_path"] + "/*.*")
        train_hr_paths = glob.glob(self.options["train_hr_dataset_path"] + "/*.*")

        # 对路径列表进行排序
        train_lr_paths = sorted(train_lr_paths)
        train_lr_reg_paths = sorted(train_lr_reg_paths)
        degrade_hr_paths = sorted(degrade_hr_paths)
        train_hr_paths = sorted(train_hr_paths)

        train_dataloader = DataLoader(
            ImageDataset(train_lr_paths, degrade_hr_paths, train_hr_paths, train_lr_reg_paths),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_cpu)  # ONLY LOAD HALF OF CPU AVAILABLE
        dataset_length = len(os.listdir(self.options["train_hr_dataset_path"]))

        # Check if we need to load weight
        if self.args.auto_resume_best or self.args.auto_resume_closest:
            self.load_weight(self.model_name)
        elif self.args.pretrained_path != "":  # If we give a pretrained path, we will use it (Should have in GAN training which uses pretrained L1 loss Network)
            self.load_pretrained(self.model_name)

        # Start iterating the epochs
        start_epoch = self.start_iteration // math.ceil(dataset_length / self.options['train_batch_size'])
        n_epochs = self.n_iterations // math.ceil(dataset_length / self.options['train_batch_size'])
        iteration_idx = self.start_iteration  # init the iteration index
        self.batch_idx = iteration_idx
        self.adjust_learning_rate(iteration_idx)  # adjust the learning rate to the desired one at the beginning

        for epoch in range(start_epoch, n_epochs):
            print("This is epoch {} and the start iteration is {} with learning rate {}".format(epoch, iteration_idx,
                                                                                                self.optimizer_g.param_groups[
                                                                                                    0]['lr']))

            # Generate new lr degradation image
            if epoch != start_epoch and epoch % self.options['degradate_generation_freq'] == 0:
                # pass
                t1 = 1  # pairs --> grl
                self.generate_lr(t1)
                self.generate_lr(t1 + 1)

            # Batch training
            loss_per_epoch = 0.0
            self.generator.train()
            tqdm_bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch_idx, imgs in enumerate(tqdm_bar):

                imgs_lr = imgs["lr"].cuda()
                imgs_lr_reg = imgs["lr_reg"].cuda()
                imgs_degrade_hr = imgs["degrade_hr"].cuda()
                imgs_hr = imgs["hr"].cuda()

                # print(imgs_lr.shape)
                # print(imgs_degrade_hr.shape)
                # print(imgs_hr.shape)

                # Used for each iteration
                self.generator_loss = 0
                self.single_iteration(imgs_lr, imgs_lr_reg, imgs_degrade_hr, imgs_hr, epoch, batch_idx)

                # tensorboard and updates
                self.tensorboard_report(iteration_idx)
                loss_per_epoch += self.generator_loss.item()

                ################################# Save model weights and update hyperparameter ########################################
                if self.lowest_generator_loss >= self.generator_loss.item():
                    self.lowest_generator_loss = self.generator_loss.item()
                    print("\nSave model with the lowest generator_loss among all iteartions ",
                          self.lowest_generator_loss)

                    # Store the best
                    self.save_weight(iteration_idx, self.model_name + "_best", self.options)

                    self.lowest_tensorboard_report(iteration_idx)

                # Update iteration and learning rate
                iteration_idx += 1
                self.batch_idx = iteration_idx
                if iteration_idx % self.options['decay_iteration'] == 0:
                    self.adjust_learning_rate(iteration_idx)  # adjust the learning rate to the desired one
                    print(
                        "Update the learning rate to {} at iteration {} ".format(self.optimizer_g.param_groups[0]['lr'],
                                                                                 iteration_idx))

                # Don't clean any memory here, it will dramatically slow down the code

            # Per epoch report
            self.tensorboard_epoch_draw(loss_per_epoch / batch_idx, epoch)

            # Per epoch store weight
            self.save_weight(iteration_idx, self.model_name + "_closest", self.options)
            # Backup Checkpoint (Per 50 epoch)
            if epoch % self.options['checkpoints_freq'] == 0 or epoch == n_epochs - 1:
                self.save_weight(iteration_idx, "checkpoints/" + self.model_name + "_epoch_" + str(epoch), self.options)

            # Clean unneeded GPU cache (since we use subprocess for generate_lr(), so we need to kill them all)
            torch.cuda.empty_cache()
            time.sleep(5)  # For enough time to clean the cache

    def single_iteration(self, imgs_lr, imgs_lr_reg, imgs_degrade_hr, imgs_hr, epoch, idx):

        ############################################# Generator section ##################################################
        self.optimizer_g.zero_grad()
        if self.has_discriminator:
            for p in self.discriminator.parameters():
                p.requires_grad = False

        # 正则化数据预处理
        imgs_lr = torch.cat((imgs_lr, imgs_lr_reg), 0)  # 2B C H W
        imgs_degrade_hr = imgs_degrade_hr.repeat(2, 1, 1, 1)
        imgs_hr = imgs_hr.repeat(2, 1, 1, 1)

        with torch.cuda.amp.autocast():
            # generate high res image
            gen_hr, net_g_out = self.generator(imgs_lr)
            # print(f"====>gen_hr.shape={gen_hr.shape}")  # [64, 3, 256, 256]  B C H W
            # print(f"====>imgs_lr.shape={imgs_lr.shape}")
            # print(f"====>imgs_hr.shape={imgs_hr.shape}")
            # print(f"====>imgs_degrade_hr.shape={imgs_degrade_hr.shape}")
            # imgs_lr 和 imgs_lr_reg 拼接

            # print(gen_hr.shape)
            if idx % 1200 == 0:
                save_image(gen_hr, f'check_M4_S2/epoch_{epoch}_genhr.png')
                save_image(imgs_lr, f'check_M4_S2/epoch_{epoch}_imgslr.png')
                save_image(imgs_hr, f'check_M4_S2/epoch_{epoch}_imgshr.png')
            # all distinct loss will be stored in self.weight_store (per iteration)
            self.calculate_loss(gen_hr, imgs_degrade_hr)

            # print(self.generator_loss)

            # reg_loss
            out = torch.nn.functional.adaptive_avg_pool2d(net_g_out, 1)
            out = out.reshape(out.shape[0], out.shape[1], -1).permute(0, 2, 1)
            x1, x2 = torch.chunk(out, 2, 0)
            x1 = x1 - x1.mean(0, keepdims=True)
            x2 = x2 - x2.mean(0, keepdims=True)
            l_reg, n_reg = torch.tensor(0).to(out), torch.tensor(0).to(out)
            for u, v in zip(x1, x2):
                out_ = torch.cat([u, v])
                nolinear_out = random_fourier_features_gpu(out_).reshape(out_.shape[0],
                                                                         -1)  # random_fourier_features_gpu(out_).shape====>torch.Size([2, 3, 1])
                u_1 = u.detach().cpu().numpy().flatten()
                v_1 = v.detach().cpu().numpy().flatten()
                wave_similarity = 1 - wavelet_similarity(u_1 , v_1)  # [0,1]==> 1-wave_similarity=0.04667052626609802
                # print(f"wave_similarity====>{wave_similarity}")
                # print(f"u===>{u_1}")
                # print(f"v===>{v_1}")
                # print(f"wave_similarity===>{wave_similarity}")
                a, b = torch.chunk(nolinear_out, 2, 0)
                l_reg += 10 * reg(u, v)
                n_reg += 10 * reg(a, b) + wave_similarity
                # n_reg += wave_similarity * 0.01

                # n_reg += 10 * wave_similarity
                # print(f"x1.shape===>{x1.shape}++++++x2.shape===>{x2.shape}")
                # print(f"u.shape===>{u.shape}++++++v.shape===>{v.shape}")
                # print(f"out_.shape===>{out_.shape}++++++nolinear_out.shape===>{nolinear_out.shape}")
                # print(f"a.shape===>{a.shape}++++++b.shape===>{b.shape}")
                # print(f"l_reg===>{l_reg}++++++n_reg===>{n_reg}")
            self.generator_loss += l_reg
            self.generator_loss += n_reg

            # print(self.generator_loss)

        # backward needed loss
        # self.loss_generator_total.backward()
        # self.optimizer_g.step()
        scaler.scale(self.generator_loss).backward()  # loss backward
        scaler.step(self.optimizer_g)
        scaler.update()
        ###################################################################################################################

        if self.has_discriminator:
            ##################################### Discriminator section  #####################################################
            for p in self.discriminator.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()

            # discriminator real input
            with torch.cuda.amp.autocast():
                # We only need imgs_degrade_hr instead of imgs_hr in discriminator (Thus, we don't want to introduce usm in the discriminator)
                real_d_preds = self.discriminator(imgs_degrade_hr)
                l_d_real = self.cri_gan(real_d_preds, True, is_disc=True)
            scaler.scale(l_d_real).backward()

            # discriminator fake input
            with torch.cuda.amp.autocast():
                fake_d_preds = self.discriminator(gen_hr.detach().clone())
                l_d_fake = self.cri_gan(fake_d_preds, False, is_disc=True)
            scaler.scale(l_d_fake).backward()

            # update
            scaler.step(self.optimizer_d)
            scaler.update()
            ##################################################################################################################

    def load_pretrained(self, name):
        # This part will load generator weight here, and it doesn't need to

        weight_dir = self.args.pretrained_path
        if not os.path.exists(weight_dir):
            print("No such pretrained " + weight_dir + " file exists! We end the program! Please check the dir!")
            os._exit(0)

        checkpoint_g = torch.load(weight_dir)
        if 'model_state_dict' in checkpoint_g:
            self.generator.load_state_dict(checkpoint_g['model_state_dict'])
        elif 'params_ema' in checkpoint_g:
            self.generator.load_state_dict(checkpoint_g['params_ema'])
        else:
            raise NotImplementedError("We didn't cannot locate the weight of thie pretrained weight")

        print(f"We will use pretrained " + name + " weight!")

    def load_weight(self, head_prefix):
        # Resume best or the closest weight available
        head = head_prefix + "_best" if self.args.auto_resume_best else head_prefix + "_closest"

        if os.path.exists("saved_models/" + head + "_generator.pth"):
            print("We need to resume previous " + head + " weight")

            # Generator
            checkpoint_g = torch.load("saved_models/" + head + "_generator.pth")
            self.generator.load_state_dict(checkpoint_g['model_state_dict'])
            self.optimizer_g.load_state_dict(checkpoint_g['optimizer_state_dict'])

            # Discriminator
            if self.has_discriminator:
                checkpoint_d = torch.load("saved_models/" + head + "_discriminator.pth")
                self.discriminator.load_state_dict(checkpoint_d['model_state_dict'])
                self.optimizer_d.load_state_dict(checkpoint_d['optimizer_state_dict'])
                assert (checkpoint_g['iteration'] == checkpoint_d[
                    'iteration'])  # must be the same for iteration in generator and discriminator

            self.start_iteration = checkpoint_g['iteration'] + 1

            # Prepare lowest generator
            if os.path.exists("saved_models/" + head_prefix + "_best_generator.pth"):
                checkpoint_g = torch.load(
                    "saved_models/" + head_prefix + "_best_generator.pth")  # load generator weight
            else:
                print("There is no best weight exists!")
            self.lowest_generator_loss = min(self.lowest_generator_loss, checkpoint_g["lowest_generator_weight"])
            print("The lowest generator loss at the beginning is ", self.lowest_generator_loss)
        else:
            print(
                f"No saved_models/" + head + "_generator.pth " or " saved_models/" + head + "_discriminator.pth exists")

        print(f"We will start from the iteration {self.start_iteration}")

    def save_weight(self, iteration, name, opt):

        # Generator
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer_g.state_dict(),
            'lowest_generator_weight': self.lowest_generator_loss,
            'opt': opt,
        }, "saved_models/" + name + "_generator.pth")
        # 'pixel_loss': self.weight_store["pixel_loss"],
        # 'perceptual_loss': self.weight_store['perceptual_loss'],
        # 'gan_loss': self.weight_store["gan_loss"],

        if self.has_discriminator:
            # Discriminator
            torch.save({
                'iteration': iteration,
                'model_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': self.optimizer_d.state_dict(),
            }, "saved_models/" + name + "_discriminator.pth")

    def lowest_tensorboard_report(self, iteration):
        self.writer.add_scalar('Loss/lowest-weight', self.generator_loss, iteration)

    @torch.no_grad()
    def generate_lr(self, t1):
        # Assert check
        if t1 == 1:
            flag = 'no'
        else:
            flag = 'yes'
        # If we directly use API, pytorch2.0 may raise an unknown bugs which is extremely slow on degradation pipeline
        print(f"python scripts/generate_lr_esr.py --flag {flag}")
        os.system(f"python scripts/generate_lr_esr.py --flag {flag}")

        # Assert check
        if t1 == 1:
            lr_paths = os.listdir(self.options["lr_dataset_path"])
        else:
            lr_paths = os.listdir(self.options["lr_dataset_path_reg"])

        degrade_hr_paths = os.listdir(self.options["degrade_hr_dataset_path"])
        hr_paths = os.listdir(self.options["train_hr_dataset_path"])

        assert (len(lr_paths) == len(degrade_hr_paths))
        assert (len(lr_paths) == len(hr_paths))
