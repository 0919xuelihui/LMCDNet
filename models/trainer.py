import numpy as np
import matplotlib.pyplot as plt
import os

import utils
from models.LightBiT import *

import torch
import torch.optim as optim
from PIL import Image

from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy, weighted_cross_entropy, MixLoss, boundary_loss, FocalLoss, BDEnhancedCELoss
from models.lr_scheduler import *

from misc.logger_tool import Logger, Timer

from utils import de_norm

palette = [0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


# 设置学习率衰减策略 ——> 每epoch更新
##### No WarmUp
def get_scheduler(optimizer, args):
    """Return a learning rate scheduler
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)  # lr_lambda参数应为计算lambda数值的函数或函数List
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3  # 1 => 1/100
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif args.lr_policy == 'multistep':
        milestones = [40, args.max_epochs//3 * 2]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif args.lr_policy == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


#### WarmUp ——> 每batch更新
def get_scheduler_warmup(optimizer, args, max_iters, warmup_iters):
    """Return a learning rate scheduler
        with warmup mechanism
    """
    milestones = [(max_iters - warmup_iters) // 3, (max_iters - warmup_iters) // 3 * 2]
    if args.warmup_lr_policy == 'poly':
        scheduler = WarmupPolyLrScheduler(optimizer, power=0.9,
                                         max_iter=max_iters, warmup_iter=warmup_iters,
                                         warmup_ratio=0.1, warmup='exp', last_epoch=-1, )
    elif args.warmup_lr_policy == 'exp':
        scheduler = WarmupExpLrScheduler(optimizer, gamma=0.98, interval=15, warmup_iter=warmup_iters,
                                          warmup_ratio=0.1, warmup='exp', last_epoch=-1, )
    elif args.warmup_lr_policy == 'step':
        scheduler = WarmupStepLrScheduler(optimizer, milestones=milestones, gamma=0.1, warmup_iter=warmup_iters,
                                          warmup_ratio=0.1, warmup='exp', last_epoch=-1, )
    elif args.warmup_lr_policy == 'linear':
        scheduler = WarmupLinearLrScheduler(optimizer, max_iter=max_iters, warmup_iter=warmup_iters,
                                            warmup_ratio=0.1, warmup='exp', last_epoch=-1, )
    else:
        return NotImplementedError('learning rate policy [%s] with warmup mechanism is not implemented',
                                   args.warmup_lr_policy)
    return scheduler


# 设置优化器(只对卷积层的相关参数添加weight decay)
# def get_optimizer(model, args):
#     wd_params, non_wd_params = [], []
#     for name, param in model.named_parameters():
#         if param.dim() == 1:
#             non_wd_params.append(param)
#         elif param.dim() == 2 or param.dim() == 4:
#             wd_params.append(param)
#     params_list = [
#         {'params': wd_params, },
#         {'params': non_wd_params, 'weight_decay': 0},
#     ]
#     if args.optimizer == 'sgd':
#         optimizer = optim.SGD(params_list, lr=args.lr, momentum=0.99, weight_decay=args.weight_decay)
#     elif args.optimizer == 'adam':
#         optimizer = optim.Adam(params_list, lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
#     elif args.optimizer == 'adamW':
#         optimizer = optim.AdamW(params_list, lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
#     else:
#         return NotImplementedError('optimizer named [%s] is not implemented',
#                                    args.optimizer)
#     return optimizer


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
    return optimizer


class CDTrainer():

    def __init__(self, args, dataloaders):
        # Prepare the data
        self.dataloaders = dataloaders
        self.imgsize = args.img_size

        self.n_class = args.n_class
        # define Network
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")
        print(self.device)

        # Learning rate
        self.lr = args.lr
        self.lr_list = []   #  Store the values of learning rate

        # define optimizers
        self.optimizer_G = get_optimizer(self.net_G, args)

        # define lr schedulers
        self.is_warmup = args.is_warmup
        if self.is_warmup:
            self.max_iters = args.max_epochs * len(dataloaders['train'])  # warmup
            self.warmup_iters = 5 * len(dataloaders['train'])
            self.exp_lr_scheduler_G = get_scheduler_warmup(self.optimizer_G, args, self.max_iters, self.warmup_iters)
        else:
            self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)  # No warmup

        # Define Deep Supervision and Boundary Enhancement
        self.deep_sup = args.deep_sup
        self.bd_enhance = args.bd_enhance
        self.bd_coeff = args.BD_weight
        self.boundary_loss = boundary_loss

        # define the loss functions
        self.weight = torch.FloatTensor([1 - args.CELoss_weight, args.CELoss_weight])
        self.loss = args.loss
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'wce':
            self._pxl_loss = weighted_cross_entropy
        elif args.loss == 'mix':
            self._pxl_loss = MixLoss(alpha=1, beta=args.dice_weight)
        elif args.loss == 'focal':
            self._pxl_loss = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_weight)
        elif args.loss == 'bd_ce':
            self._pxl_loss = BDEnhancedCELoss(alpha=1, beta=args.BD_weight)
        else:
            raise NotImplemented(args.loss)

        # whether to use training trick
        self.trick = args.trick

        # define Metric Tool
        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()

        self.batch_size = args.batch_size
        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1)  # size : B,H,W
        return pred

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()
        self.lr_list.append(self.optimizer_G.param_groups[0]['lr'])

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        current_f1 = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        if np.mod(self.batch_id, 10) == 1:
            if self.is_training == True:
                imps, est = self._timer_update()
                message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, f1_0: %.5f, f1_1: %.5f\n' % \
                          (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.batch_id, m - 1,
                           imps * self.batch_size, est, self.G_loss.item(), current_f1[0], current_f1[1])
                self.logger.write(message)
            else:
                message = 'Is_training: %s. [%d,%d][%d,%d], f1_0: %.5f, f1_1: %.5f\n' % \
                          (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.batch_id, m - 1,
                           current_f1[0], current_f1[1])
                self.logger.write(message)

        if np.mod(self.batch_id, 10) == 1:  # store the running results every 10 batch
            if self.is_training == False:
                prediction = self._visualize_pred()[0, :, :].cpu().numpy()
                gt = self.batch['L'][0, :, :].cpu().numpy().astype(np.uint8)
                flag = gt - prediction
                prediction[flag == 1] = 3  # False Negative
                prediction[flag == -1] = 2  # False Positive
                prediction_color = colorize_mask(prediction)
                gt_color = colorize_mask(gt)
                width, height = self.imgsize, self.imgsize
                save_image = Image.new('RGB', (width * 2, height * 1))
                save_image.paste(gt_color, box=(0 * width, 0 * height))
                save_image.paste(prediction_color, box=(1 * width, 0 * height))
                file_name = os.path.join(
                    self.vis_dir, 'istrain_' + str(self.is_training) + '_' +
                                  str(self.epoch_id) + '_' + str(self.batch_id) + '.jpg')
                save_image.save(file_name)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['F1_1']
        message = 'Is_training: %s. Epoch %d / %d, epoch_F1_1= %.5f\n' % (self.is_training,
                    self.epoch_id, self.max_num_epochs-1, self.epoch_acc)
        self.logger.write(message)
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_f1_1=%.4f, Historical_best_f1_1=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _visualize_acc(self):
        #  visualize the accuracy curve
        train_path = os.path.join(self.checkpoint_dir, 'train_acc.npy')
        val_path = os.path.join(self.checkpoint_dir, 'val_acc.npy')
        train_acc = np.load(train_path)
        val_acc = np.load(val_path)
        t1 = np.arange(1, len(train_acc) + 1)
        t2 = np.arange(1, len(val_acc) + 1)
        plt.figure(1)
        plt.plot(t1, train_acc, 'r', t2, val_acc, 'b')
        label = ['train_F1', 'val_F1']
        plt.legend(label, loc='upper left')
        plt.savefig(os.path.join(self.checkpoint_dir, 'F1_vis.jpg'))
        plt.close(1)
        #  visualize the learning rate curve
        t = np.arange(1, len(self.lr_list) + 1)
        plt.figure(2)
        plt.plot(t, self.lr_list, 'b')
        label = ['learning rate']
        plt.legend(label, loc='upper left')
        plt.savefig(os.path.join(self.checkpoint_dir, 'Lr.jpg'))
        plt.close(2)

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        if self.bd_enhance == False:
            if self.deep_sup == True:
                self.G_pred, self.aux_out = self.net_G(img_in1, img_in2)
            else:
                self.G_pred = self.net_G(img_in1, img_in2)
        else:
            self.G_pred, self.bd_pred = self.net_G(img_in1, img_in2)

    def _backward_G(self):
        gt = self.batch['L'].to(self.device).long()
        if self.loss == 'bd_ce':  # 考虑边界损失的情况（先用CE训练到一定精度后，再引入边界损失在小学习率下微调）
            if self.batch_id >= 45:
                self.G_loss = self._pxl_loss(self.G_pred, gt, weight=self.weight.cuda())
            else:
                self.G_loss = cross_entropy(self.G_pred, gt, weight=self.weight.cuda())
        else:  # 一般情况
            self.G_loss = self._pxl_loss(self.G_pred, gt, weight=self.weight.cuda())
        if self.deep_sup == True:
            self.aux_loss = [cross_entropy(pred, gt, weight=self.weight.cuda()) for pred in self.aux_out]
            self.G_loss += sum(self.aux_loss)
        if self.bd_enhance == True:
            if self.epoch_id >= 45:
                self.bd_loss = self.boundary_loss(self.bd_pred, gt)
                self.G_loss += self.bd_coeff * self.bd_loss
                # if self.batch_id == 0:
                #     bd_print = self.bd_loss.detach().cpu().numpy()
                #     print(bd_print)
        self.G_loss.backward()

    def train_models(self):
        self._load_checkpoint()

        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            ################## train #################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            if self.trick == True:
                # if self.epoch_id > 64:  # Flip、No Flip结合的训练方式
                #     current_dataloader = self.dataloaders['train_flip']
                # else:
                #     current_dataloader = self.dataloaders['train_noflip']
                if self.optimizer_G.param_groups[0]['lr'] > self.lr * 0.01:  # Flip、No Flip结合的训练方式
                    current_dataloader = self.dataloaders['train_flip']
                else:
                    current_dataloader = self.dataloaders['train_noflip']
            else:
                current_dataloader = self.dataloaders['train']  # 原始训练方式
            for self.batch_id, batch in enumerate(current_dataloader, 0):
                self._forward_pass(batch)
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
                self._timer_update()
                if self.is_warmup:
                    self._update_lr_schedulers()  # For WarmUp
            self._collect_epoch_states()
            self._update_training_acc_curve()
            if not self.is_warmup:
                self._update_lr_schedulers()  # For No WarmUp

            ################## Eval ##################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            self._update_val_acc_curve()
            self._update_checkpoints()
        self._visualize_acc()
