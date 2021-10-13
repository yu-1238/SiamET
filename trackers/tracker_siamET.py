import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from ET.siamban.utils.model_load import load_pretrain
from .utils import get_subwindow_tracking,generate_anchor,get_axis_aligned_bbox,Round
from .config import TrackerConfig
from ET.net_et import ET_seq
from ET.siamban.models.model_builder import ModelBuilder
from ET.siamban.core.config  import cfg
from os.path import join
from collections import namedtuple
import cv2
Corner = namedtuple('Corner', 'x1 y1 x2 y2')
# alias
from ET.utils_upd import  Round, generate_points

BBox = Corner
Center = namedtuple('Center', 'x y w h')
class SiamETTracker:

    def __init__(self, model_path, et_path1, gpu_id, step, et_path0, et_path2):

        self.gpu_id = gpu_id
        config_file = join('../models/OTB100/', 'config.yaml')
        cfg.merge_from_file(config_file)
        self.cfg=cfg
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        self.net = ModelBuilder()  #

        self.is_deterministic = False


        self.net = load_pretrain(self.net,model_path ).cuda().eval()
        # self.net.load_state_dict(torch.load(model_path))
        with torch.cuda.device(self.gpu_id):
            self.net = self.net.cuda()
        self.net.eval()

        self.state = dict()

        self.step = step  # 1,2,3
        if self.step == 1:
            self.name = 'tracker'
        elif self.step == 2:
            self.name = 'Linear'
        elif self.step==4:
            self.name='test2'
        else:

            dataset = et_path1.split('/')[-1].split('.')[0]
            if dataset == 'vot2018' or dataset == 'vot2016':
                self.name = 'ET'
            else:
                self.name = dataset

        if self.step == 3:
            # load 1 enhance template network
            self.etnet =  ET_seq()


            et_model = torch.load(et_path1)['state_dict']
            et_model_fix = dict()
            for i in et_model.keys():
                if i.split('.')[0] == 'module':
                   et_model_fix['.'.join(i.split('.')[1:])] = et_model[i]
                else:
                    et_model_fix[i] = et_model[i]  # 单GPU模型直接赋值

            self.etnet.load_state_dict(et_model_fix)

            self.etnet.eval().cuda()
        elif self.step == 4:
            self.et1=  ET_seq()
            self.et0= ET_seq()
            self.et2= ET_seq()

            et_model = torch.load(et_path1)['state_dict']
            et_model_fix = dict()
            for i in et_model.keys():
                if i.split('.')[0] == 'module':
                    et_model_fix['.'.join(i.split('.')[1:])] = et_model[i]
                else:
                    et_model_fix[i] = et_model[i]
            self.et1.load_state_dict(et_model_fix)
            self.et1.eval().cuda()

            et_model = torch.load(et_path0)['state_dict']
            et_model_fix = dict()
            for i in et_model.keys():
                if i.split('.')[0] == 'module':
                    et_model_fix['.'.join(i.split('.')[1:])] = et_model[i]
                else:
                    et_model_fix[i] = et_model[i]
            self.et0.load_state_dict(et_model_fix)
            self.et0.eval().cuda()


            et_model = torch.load(et_path2)['state_dict']
            et_model_fix = dict()
            for i in et_model.keys():
                if i.split('.')[0] == 'module':
                    et_model_fix['.'.join(i.split('.')[1:])] = et_model[i]
                else:
                    et_model_fix[i] = et_model[i]
            self.et2.load_state_dict(et_model_fix)

            self.et2.eval().cuda()

        else:
            self.etnet = ''



    def init(self, im, init_rbox):

        state = self.state

        [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)
        # tracker init
        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])

        p = TrackerConfig
        # p.enhance(self.net.cfg)
        p.instance_size = 255
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]

        # if p.adaptive:
        #     if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
        #         p.instance_size = 287  # small object big search region
        #     else:
        #         p.instance_size = 271
        #     # python3
        #     p.score_size = (p.instance_size - p.exemplar_size) // p.total_stride + 1  # python3
        #
        # p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, p.score_size)
        p.score_size = 25
        p.points = generate_points(8, 25)
        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = Round(np.sqrt(wc_z * hc_z))  # python3

        # initialize the exemplar
        z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
        z_crop = Variable(z_crop)

        if self.step == 1:
            self.net.template(z_crop.cuda())

            # state['z_f_2'] = z_f[2].cpu().data

            # state['z_2'] = z_f[2].cpu().data
        else:
            z_f = self.net.template(z_crop.cuda())  # [1,512,6,6]
            self.net.kernel(z_f)
            state['z_f_1'] = z_f[1].cpu().data
            state['z_1'] = z_f[1].cpu().data

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
        elif p.windowing == 'uniform':
            window = np.ones((p.score_size, p.score_size))
        # window = np.tile(window.flatten(), p.anchor_num)
        window = window.flatten()

        state['p'] = p
        state['net'] = self.net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['z_f_2'] = z_f[2].cpu().data
        state['z_2'] = z_f[2].cpu().data
        state['z_0'] = z_f[0].cpu().data
        # # state['z_1'] = z_f[1].cpu().data
        # # state['z_2'] = z_f[2].cpu().data
        state['z_f_0'] = z_f[0].cpu().data
        # state['z_f_1'] = z_f[1].cpu().data
        # state['z_f_2'] = z_f[2].cpu().data
        self.state = state

        return state

    def track(self, im):

        state = self.state
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        wc_z = target_sz[1] + p.context_amount * sum(target_sz)
        hc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)  #

        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2  # python3 2020-05-13
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # extract scaled crops for search region x at previous target position
        # x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
        x_c = get_subwindow_tracking(im, target_pos, p.instance_size, Round(s_x), avg_chans)
        x_crop = Variable(x_c)

        target_pos, target_sz, score = tracker_eval(net, x_crop.cuda(), target_pos, target_sz , window, scale_z, p,im,target_pos,self.cfg)
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

        # 更新模板
        if self.step > 1:
            z_crop = Variable(
                get_subwindow_tracking(im, target_pos, p.exemplar_size, Round(s_z), avg_chans))
            z_f = self.net.template(z_crop.cuda())
            z_f_0 = z_f[0]
            z_f_1 = z_f[1]
            z_f_2 = z_f[2]
            if self.step == 2:
                zLR = 0.0102
                z_f_1_= (1 - zLR) * Variable(state['z_f_1']).cuda() + zLR * z_f_1
                # temp = np.concatenate((init, pre, cur), axis=1)
            elif self.step ==4:
                temp_0 = torch.cat((Variable(state['z_0']).cuda(), Variable(state['z_f_0']).cuda(), z_f_0), 1)
                init_inp_0 = Variable(state['z_0']).cuda()
                z_f_0_ = self.et0(temp_0, init_inp_0)  # 累积特征图
                # 1
                temp_1 = torch.cat((Variable(state['z_1']).cuda(), Variable(state['z_f_1']).cuda(), z_f_1), 1)
                init_inp_1 = Variable(state['z_1']).cuda()
                z_f_1_ = self.et1(temp_1, init_inp_1)
                # 2
                temp_2 = torch.cat((Variable(state['z_2']).cuda(), Variable(state['z_f_2']).cuda(), z_f_2), 1)
                init_inp_2 = Variable(state['z_2']).cuda()
                z_f_2_ = self.et2(temp_2, init_inp_2)
            else:
                temp_0 = torch.cat((Variable(state['z_0']).cuda(), Variable(state['z_f_0']).cuda(), z_f_0), 1)
                init_inp_0 = Variable(state['z_0']).cuda()
                z_f_0_ = self.etnet(temp_0, init_inp_0)
                # 1
                temp_1 = torch.cat((Variable(state['z_1']).cuda(), Variable(state['z_f_1']).cuda(), z_f_1), 1)
                init_inp_1 = Variable(state['z_1']).cuda()
                z_f_1_ = self.etnet(temp_1, init_inp_1)
                # 2
                temp_2 = torch.cat((Variable(state['z_2']).cuda(), Variable(state['z_f_2']).cuda(), z_f_2), 1)
                init_inp_2 = Variable(state['z_2']).cuda()
                z_f_2_ =self. etnet(temp_2, init_inp_2)
            z_f_ = [z_f_0_, z_f_1_, z_f_2_]
            state['z_f_0'] = z_f_0_.cpu().data
            state['z_f_1'] = z_f_1_.cpu().data
            state['z_f_2'] = z_f_2_.cpu().data# 累积模板
            self.net.kernel(z_f_)  # 更新模板

        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['score'] = score
        self.state = state
        return state


def _convert_bbox(delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

def _bbox_clip(cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

def _convert_score(score):
        cls_out_channels = 2
        if cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

def corner2center(corner):
        """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
        Args:
            conrner: Corner or np.array (4*N)
        Return:
            Center or np.array (4 * N)
        """
        if isinstance(corner, Corner):
            x1, y1, x2, y2 = corner
            return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
        else:
            x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
            x = (x1 + x2) * 0.5
            y = (y1 + y2) * 0.5
            w = x2 - x1
            h = y2 - y1
            return x, y, w, h
def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p, img, center_pos,cfg):
        # delta, score = net(x_crop)
        outputs = net.track(x_crop)

        score = _convert_score(outputs['cls'])
        pred_bbox = _convert_bbox(outputs['loc'], p.points)

        # delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
        # score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()
        #
        # delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
        # delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
        # delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
        # delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # # size penalty
        # s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
        # r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty
        #
        # penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
        # pscore = penalty * score
        #
        # # window float
        # pscore = pscore * (1 - p.window_influence) + window * p.window_influence
        # best_pscore_id = np.argmax(pscore)
        #
        # target = delta[:, best_pscore_id] / scale_z
        # target_sz = target_sz / scale_z
        # lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr
        #
        # res_x = target[0] + target_pos[0]
        # res_y = target[1] + target_pos[1]
        #
        # res_w = target_sz[0] * (1 - lr) + target[2] * lr
        # res_h = target_sz[1] * (1 - lr) + target[3] * lr
        #
        # target_pos = np.array([res_x, res_y])
        # target_sz = np.array([res_w, res_h])
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(target_sz[0] * scale_z, target_sz[1] * scale_z)))  # selfsize demo frames box's w,j

        # aspect ratio penalty
        r_c = change((target_sz[0] / target_sz[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        # penalty_k = 0.055
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        ## Scale penalty
        # __C.TRACK.PENALTY_K = 0.14

        # Window influence
        # __C.TRACK.WINDOW_INFLUENCE = 0.45
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 window * cfg.TRACK.WINDOW_INFLUENCE  # 0.45

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        # __C.TRACK.LR = 0.30
        cx = bbox[0] + target_pos[0]
        cy = bbox[1] + target_pos[1]

        # smooth bbox
        width = target_sz[0] * (1 - lr) + bbox[2] * lr
        height = target_sz[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = _bbox_clip(cx, cy, width,
                                           height, img.shape[:2])
        target_pos = np.array([cx, cy])
        target_sz = np.array([width, height])

        return target_pos, target_sz, score[best_idx]
def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
        """
        args:
                    im: bgr based image
                    pos: center position
                    model_sz: exemplar size
                    s_z: original size
                    avg_chans: channel average
                """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)

        im_patch = im_patch.cuda()
        return im_patch