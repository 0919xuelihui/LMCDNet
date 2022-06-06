import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from models.LightBiT import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils


# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


palette = [0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader
        self.imgsize = args.img_size

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # Define Deep Supervision and Boundary Enhancement
        self.deep_sup = args.deep_sup
        self.bd_enhance = args.bd_enhance

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.test_dir = args.test_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.test_dir) is False:
            os.mkdir(self.test_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']
            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1)
        return pred


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

        # 计算变化、非变化区域分类的准确性
        gt = self.batch['L'].to(self.device).detach().cpu().numpy()  # 真值图（二值），size: B,H,W
        pred = torch.argmax(self.G_pred.detach(), dim=1).cpu().numpy()  # 预测图（二值），size: B,H,W
        gt = np.sum(gt, axis=(1, 2))  # size:B
        gt = (gt > 0).astype(int)
        pred = np.sum(pred, axis=(1, 2))  # size:B
        pred = (pred > 0).astype(int)
        self.cm += np.bincount(2 * gt + pred, minlength=4).reshape(2, 2)  # 分别代表TN, FP, FN, TP

        m = len(self.dataloader)

        if np.mod(self.batch_id, 3) == 1:
            message = 'Is_training: %s. [%d,%d], f1_0: %.5f, f1_1: %.5f\n' % \
                      (self.is_training, self.batch_id, m - 1, current_f1[0], current_f1[1])
            self.logger.write(message)

        if self.is_training == False:
            prediction = self._visualize_pred()[0, :, :].cpu().numpy()
            gt = self.batch['L'][0, :, :].cpu().numpy()
            flag = gt - prediction
            prediction[flag == 1] = 3  # False Negative
            prediction[flag == -1] = 2  # False Positive
            prediction_color = colorize_mask(prediction)
            gt_color = colorize_mask(gt)
            img_before = self.batch['A'][0].cpu().permute(1, 2, 0).numpy()
            img_before = (img_before * 255).astype(np.uint8)
            img_after = self.batch['B'][0].cpu().permute(1, 2, 0).numpy()
            img_after = (img_after * 255).astype(np.uint8)
            img_before = Image.fromarray(img_before).convert('RGB')
            img_after = Image.fromarray(img_after).convert('RGB')
            width, height = self.imgsize, self.imgsize
            save_image = Image.new('RGB', (width * 2, height * 2))
            save_image.paste(gt_color, box=(0 * width, 0 * height))
            save_image.paste(prediction_color, box=(1 * width, 0 * height))
            save_image.paste(img_before, box=(0 * width, 1 * height))
            save_image.paste(img_after, box=(1 * width, 1 * height))
            # file_name = os.path.join(self.test_dir, '%d.jpg' % self.batch_id)
            file_name = os.path.join(self.test_dir, self.batch['name'][0].split('.')[0] + '_%d.jpg' % self.batch_id)
            save_image.save(file_name)

    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['F1_1']
        message = 'Test: epoch_F1_1= %.5f\n' % (self.epoch_acc)
        self.logger.write(message)

        self.Precise_nc = self.cm[0, 0] / (self.cm[0, 0] + self.cm[1, 0])
        self.Recall_nc = self.cm[0, 0] / (self.cm[0, 0] + self.cm[0, 1])
        message = 'Test: Pre of Nochange Area = %.5f\nRecall of Nochange Area = %.5f\n' % (self.Precise_nc, self.Recall_nc)
        self.logger.write(message)

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

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
                # self.G_pred = self.net_G(img_in1, img_in2, self.batch['name'][0].split('.')[0])
        else:
            self.G_pred, self.bd_pred = self.net_G(img_in1, img_in2)

    def eval_models(self, checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Test ##################
        ##########################################
        self.logger.write('Begin Test...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()
        self.cm = np.array([[0, 0], [0, 0]])  # 分别代表TN, FP, FN, TP

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
        print(self.cm)
