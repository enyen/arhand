import sys
import cv2
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import acr.config as config
from acr.model import ACR as ACR_v1
from acr.visualization import Visualizer
from acr.mano_wrapper import MANOWrapper
from acr.config import args, parse_args, ConfigContext
from acr.utils import (justify_detection_state, reorganize_results, img_preprocess, WebcamVideoStream,
                       create_OneEuroFilter, load_model, smooth_results, get_remove_keys)


class ACR(nn.Module):
    def __init__(self, args_set=None):
        super(ACR, self).__init__()
        self.demo_cfg = {'mode': 'parsing', 'calc_loss': False}
        self.project_dir = config.project_dir
        self._initialize_(vars(args() if args_set is None else args_set))
        self.visualizer = Visualizer(resolution=(self.render_size, self.render_size), renderer_type=self.renderer)
        self._build_model_()

    def _initialize_(self, config_dict):
        # configs
        hparams_dict = {}
        for i, j in config_dict.items():
            setattr(self, i, j)
            hparams_dict[i] = j

        # optimizations parameters
        if self.temporal_optimization:
            self.filter_dict = {}
            self.filter_dict[0] = create_OneEuroFilter(args().smooth_coeff)
            self.filter_dict[1] = create_OneEuroFilter(args().smooth_coeff)

        return hparams_dict

    def _build_model_(self):
        model = ACR_v1().eval()
        model = load_model(self.model_path, model, prefix='module.', drop_prefix='', fix_loaded=False)
        self.model = model.to(args().device)
        self.model.eval()
        self.mano_regression = MANOWrapper().to(args().device)

    @torch.no_grad()
    def process_results(self, outputs):
        # temporal optimization
        if self.temporal_optimization:
            out_hand = []  # [0],[1],[0,1]
            for idx, i in enumerate(outputs['detection_flag_cache']):
                if i:
                    out_hand.append(idx)  # idx is also hand type, 0 for left, 1 for right
                else:
                    out_hand.append(-1)

            assert len(outputs['params_dict']['poses']) == 2
            for sid, tid in enumerate(out_hand):
                if tid == -1:
                    continue
                outputs['params_dict']['poses'][sid], outputs['params_dict']['betas'][sid] = (
                    smooth_results(self.filter_dict[tid], outputs['params_dict']['poses'][sid], outputs['params_dict']['betas'][sid]))

        outputs = self.mano_regression(outputs, outputs['meta_data'])
        reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
        new_results = reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)

        return outputs, new_results

    @torch.no_grad()
    def forward(self, bgr_frame, path):
        with torch.no_grad():
            outputs = self.single_image_forward(bgr_frame, path)

        if outputs is not None and outputs['detection_flag']:
            outputs, results = self.process_results(outputs)

            # visualization: render to raw image
            show_items_list = ['mesh']
            results_dict, img_names = self.visualizer.visulize_result_live(outputs, bgr_frame, outputs['meta_data'],
                                                                           show_items=show_items_list,
                                                                           vis_cfg={'settings': ['put_org']},
                                                                           save2html=False)
            img_name, mesh_rendering_orgimg = img_names[0], results_dict['mesh_rendering_orgimgs']['figs'][0]
            cv2.imshow('render_output', mesh_rendering_orgimg[:, :, ::-1])
        else:
            results = {}
            results[path] = {}
            cv2.imshow('render_output', bgr_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            return -1

        return results

    @torch.no_grad()
    def single_image_forward(self, bgr_frame, path):
        meta_data = img_preprocess(bgr_frame, path, input_size=args().input_size, single_img_input=True)

        ds_org, imgpath_org = get_remove_keys(meta_data, keys=['data_set', 'imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.model_precision == 'fp16':
            with autocast():
                outputs = self.model(meta_data, **self.demo_cfg)
        else:
            outputs = self.model(meta_data, **self.demo_cfg)

        outputs['detection_flag'], outputs['reorganize_idx'] = justify_detection_state(outputs['detection_flag'],
                                                                                       outputs['reorganize_idx'])
        meta_data.update({'imgpath': imgpath_org, 'data_set': ds_org})
        outputs['meta_data']['imgpath'] = [path]

        return outputs


if __name__ == '__main__':
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        acr = ACR(args_set=args_set)

    cap = WebcamVideoStream(args().cam_id)
    cap.start()
    while True:
        frame = cap.read()
        outputs = acr(frame, '0')
        if outputs == -1:
            break

    cap.stop()
