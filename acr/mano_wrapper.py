import torch
import torch.nn as nn
from os import path
from acr.config import args
from acr.utils import vertices_kp3d_projection
from manotorch.manolayer import ManoLayer


class MANOWrapper(nn.Module):
    def __init__(self, asset_rel_path):
        super().__init__()
        dir_assets = path.join(path.dirname(__file__), asset_rel_path)

        self.mano_layer = nn.ModuleDict({
            'r': ManoLayer(
                ncomps=45,
                center_idx=args().align_idx if args().mano_mesh_root_align else None,
                side='right',
                mano_assets_root=dir_assets,
                use_pca=False,
                flat_hand_mean=False,
            ),
            'l': ManoLayer(
                ncomps=45,
                center_idx=args().align_idx if args().mano_mesh_root_align else None,
                side='left',
                mano_assets_root=dir_assets,
                use_pca=False,
                flat_hand_mean=False,
            )
        })
        self.mano_layer['l'].th_shapedirs[:, 0, :] *= -1

    def forward(self, outputs, meta_data, depth, focal_length):
        params_dict = outputs['params_dict']
        L, R = outputs['left_hand_num'], outputs['right_hand_num']
        outputs['output_hand_type'] = torch.cat((torch.zeros(L), torch.ones(R))).to(torch.int32)

        handl = self.mano_layer['l'](params_dict['poses'][:L], params_dict['betas'][:L])
        l_vertices, l_joints = handl.verts, handl.joints
        handr = self.mano_layer['r'](params_dict['poses'][L:L + R], params_dict['betas'][L:L + R])
        r_vertices, r_joints = handr.verts, handr.joints

        mano_outs = {'verts': torch.cat((l_vertices, r_vertices)), 'j3d': torch.cat((l_joints, r_joints))}
        outputs.update({**mano_outs})
        outputs.update(vertices_kp3d_projection(outputs, params_dict=params_dict, depth=depth,
                                                focal_length=focal_length, meta_data=meta_data))

        return outputs
