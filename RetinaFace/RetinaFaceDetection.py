from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from RetinaFace.data import cfg_mnet, cfg_re50
from RetinaFace.layers.functions.prior_box import PriorBox
from RetinaFace.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from RetinaFace.models.retinaface import RetinaFace
from RetinaFace.utils.box_utils import decode, decode_landm
import time
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class retina_face:
    def __init__(self,crop_size, args):
        self.args = args
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu_num
        # -------------------------- load model ------------------------------------
        torch.set_grad_enabled(False)
        self.cfg = None
        if self.args.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif self.args.network == "resnet50":
            self.cfg = cfg_re50
        # net and model
        net = RetinaFace(cfg=self.cfg, phase = 'test')
        net = load_model(net, self.args.trained_model, self.args.cpu)
        net.eval()
        # print('Finished loading model!')
        # print(net)
        cudnn.benchmark = True
        self.device = torch.device("cpu" if self.args.cpu else "cuda")
        self.net = net.to(self.device)

        self.resize = 1
        self.crop_size = crop_size
        # print('Retina align start!')

    def detect_face(self, image_path):
        if type(image_path) == str :
            frame = cv2.imread(image_path)
        else:
            frame = image_path
        face = frame
        img_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        # tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        # print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.args.keep_top_k, :]
        landms = landms[:self.args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        output_raw = img_raw.copy()
        output_det = []
        output_det_draw = []
        output_points = []
        output_points_crop = []
        bbox = []
        # show image
        if self.args.save_image:
            for i,b in enumerate(dets):
                if b[4] < self.args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                if float(text) >= 0.95 and b[0] > 0 and b[1] > 0 and b[2] > 0 and b[3] > 0:
                    img_det = img_raw[b[1]:b[3], b[0]:b[2], :].copy()
                    
                    img_det = cv2.resize(img_det,(self.crop_size, self.crop_size))
                    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cv2.rectangle(face, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cx = b[0]
                    cy = b[1] + 25
                    # landmarks
                    cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4) # Ojo izquierdo
                    cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4) # Ojo derecho
                    cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4) # Nariz
                    cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4) # Lado izquierdo de la boca
                    cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4) # Lado derecho de la boca
                    det_draw = img_raw[b[1]:b[3], b[0]:b[2], :].copy()
                    det_draw = cv2.resize(det_draw,(self.crop_size, self.crop_size))

                    scale_x = img_raw.shape[1] / self.crop_size
                    scale_y = img_raw.shape[0] / self.crop_size
                    # save image
                    img_det = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
                    det_draw = cv2.cvtColor(det_draw, cv2.COLOR_BGR2RGB)

                    output_det.append(img_det)
                    output_det_draw.append(det_draw)
                    output_points.append([[b[5],b[6]],[b[7],b[8]],[b[9],b[10]],[b[11],b[12]],[b[13],b[14]]])
                    output_points_crop.append([[b[5] - b[0],b[6] - b[1]],[b[7] - b[0],b[8] - b[1]],[b[9] - b[0],b[10] - b[1]],[b[11] - b[0],b[12] - b[1]],[b[13] - b[0],b[14] - b[1]]])
                    bbox.append([b[0],b[1],b[2],b[3]])
            output_raw = cv2.cvtColor(output_raw, cv2.COLOR_BGR2RGB)
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        return img_raw,output_raw, output_points,bbox,output_det_draw
        #return img_raw,output_raw, output_points,bbox
