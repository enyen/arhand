import os
import cv2
import logging
import numpy as np
from .config import args

import torch
import torch.nn.functional as F
import imgaug.augmenters as iaa
from imgaug.augmenters import compute_paddings_to_reach_aspect_ratio
from threading import Thread


def BHWC_to_BCHW(x):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    return x.unsqueeze(1).transpose(1, -1).squeeze(-1)


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        hom_mat = torch.tensor([0, 0, 1]).float()
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        batch_size, device = rot_mat.shape[0], rot_mat.device
        hom_mat = hom_mat.view(1, 3, 1)
        hom_mat = hom_mat.repeat(batch_size, 1, 1).contiguous()
        hom_mat = hom_mat.to(device)
        rotation_matrix = torch.cat([rot_mat, hom_mat], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def rot6d_to_rotmat(x):
    x = x.view(-1, 3, 2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats


def rot6D_to_angular(rot6D):
    batch_size = rot6D.shape[0]
    pred_rotmat = rot6d_to_rotmat(rot6D).view(batch_size, -1, 3, 3)
    pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(batch_size, -1)
    return pose


def batch_orth_proj(X, camera, mode='2d', keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:, :, :2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:, :, 2].unsqueeze(-1)], -1)
    return X_camed


def convert_kp2d_from_input_to_orgimg(kp2ds, offsets):
    offsets = offsets.float().to(kp2ds.device)
    img_pad_size, crop_trbl, pad_trbl = offsets[:, :2], offsets[:, 2:6], offsets[:, 6:10]
    leftTop = torch.stack([crop_trbl[:, 3] - pad_trbl[:, 3], crop_trbl[:, 0] - pad_trbl[:, 0]], 1)
    kp2ds_on_orgimg = (kp2ds + 1) * img_pad_size.unsqueeze(1) / 2 + leftTop.unsqueeze(1)
    return kp2ds_on_orgimg


def vertices_kp3d_projection(outputs, params_dict, meta_data=None, depth=None):
    params_dict, vertices, j3ds = params_dict, outputs['verts'], outputs['j3d']
    verts_camed = batch_orth_proj(vertices, params_dict['cam'], mode='3d', keep_dim=True)
    pj3d = batch_orth_proj(j3ds, params_dict['cam'], mode='2d')
    predicts_pj2ds = (pj3d[:, :, :2][:, :24].detach().cpu().numpy() + 1) * 256

    # predicts_j3ds = j3ds[:, :24].contiguous().detach().cpu().numpy()
    # cam_trans = estimate_translation(predicts_j3ds, predicts_pj2ds, focal_length=args().focal_length,
    #                                  img_size=np.array([512, 512])).to(vertices.device)

    cam_trans_l = compute_3d_offset((verts_camed[0].detach().cpu().numpy() + 1) * 256,
                                    vertices[0].detach().cpu().numpy(),
                                    predicts_pj2ds[0],
                                    depth, args().focal_length, np.array([512, 512]))
    cam_trans_r = compute_3d_offset((verts_camed[1].detach().cpu().numpy() + 1) * 256,
                                    vertices[1].detach().cpu().numpy(),
                                    predicts_pj2ds[1],
                                    depth, args().focal_length, np.array([512, 512]))
    cam_trans = torch.from_numpy(np.stack((cam_trans_l, cam_trans_r))).to('cuda')

    projected_outputs = {'verts_camed': verts_camed, 'pj2d': pj3d[:, :, :2], 'cam_trans': cam_trans}

    if meta_data is not None:
        projected_outputs['pj2d_org'] = convert_kp2d_from_input_to_orgimg(projected_outputs['pj2d'],
                                                                          meta_data['offsets'])
    return projected_outputs


def compute_3d_offset(pred_vertices_img, pred_vertices_smpl, pred_joints_img, depth, focal_length, img_size):
    """
    source:
     https://github.com/yzqin/dex-hand-teleop/blob/3f7b56deed878052ec733a32b503aceee4ca8c8c/hand_detector/hand_monitor.py#L102
    """
    height, width = depth.shape
    # Image space vertices
    mask_int = np.rint(pred_vertices_img[:, :2]).astype(int)
    mask_int = np.clip(mask_int, [0, 0], [width - 1, height - 1])
    depth_vertices = depth[mask_int[:, 1], mask_int[:, 0]]
    depth_median = np.nanmedian(depth_vertices)
    depth_valid_mask = np.nonzero(np.abs(depth_vertices - depth_median) < 0.2)[0]
    valid_vertex_depth = depth_vertices[depth_valid_mask]

    # Hand frame vertices
    v_smpl = pred_vertices_smpl[depth_valid_mask]
    z_smpl = v_smpl[:, 2]
    z_near_to_far_order = np.argsort(z_smpl)

    # Filter depth with same pixel pos to the front position
    valid_mask_int = mask_int[depth_valid_mask, :][z_near_to_far_order, :]
    mask_int_encoding = valid_mask_int[:, 0] * 1e5 + valid_mask_int[:, 1]
    _, unique_indices = np.unique(mask_int_encoding, return_index=True)
    front_indices = z_near_to_far_order[unique_indices]

    # Calculate mean depth from image space and hand frame
    mean_depth_image = np.mean(valid_vertex_depth[front_indices])
    mean_depth_smpl = np.mean(z_smpl[front_indices])
    depth_offset = mean_depth_image - mean_depth_smpl

    offset_img = pred_joints_img[args().align_idx, 0:2] - img_size / 2
    offset = np.concatenate([offset_img / focal_length * depth_offset, [depth_offset]])

    return offset


def estimate_translation_cv2(joints_3d, joints_2d, focal_length=600, img_size=np.array([512., 512.]), proj_mat=None,
                             cam_dist=None):
    if proj_mat is None:
        camK = np.eye(3)
        camK[0, 0], camK[1, 1] = focal_length, focal_length
        camK[:2, 2] = img_size // 2
    else:
        camK = proj_mat
    ret, rvec, tvec, inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist, \
                                                  flags=cv2.SOLVEPNP_EPNP, reprojectionError=20, iterationsCount=100)

    if inliers is None:
        return INVALID_TRANS
    else:
        tra_pred = tvec[:, 0]
        return tra_pred


def estimate_translation_np(joints_3d, joints_2d, joints_conf, focal_length=600, img_size=np.array([512., 512.]),
                            proj_mat=None):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = joints_3d.shape[0]
    if proj_mat is None:
        # focal length
        f = np.array([focal_length, focal_length])
        # optical center
        center = img_size / 2.
    else:
        f = np.array([proj_mat[0, 0], proj_mat[1, 1]])
        center = proj_mat[:2, 2]

    # transformations
    Z = np.reshape(np.tile(joints_3d[:, 2], (2, 1)).T, -1)
    XY = np.reshape(joints_3d[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array([F * np.tile(np.array([1, 0]), num_joints), F * np.tile(np.array([0, 1]), num_joints),
                  O - np.reshape(joints_2d, -1)]).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(joints_3d, joints_2d, pts_mnum=4, focal_length=600, proj_mats=None, cam_dists=None,
                         img_size=np.array([512., 512.])):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (B, K, 3) 3D joint locations
        joints: (B, K, 2) 2D joint coordinates
    Returns:
        (B, 3) camera translation vectors
    """
    if torch.is_tensor(joints_3d):
        joints_3d = joints_3d.detach().cpu().numpy()
    if torch.is_tensor(joints_2d):
        joints_2d = joints_2d.detach().cpu().numpy()

    if joints_2d.shape[-1] == 2:
        joints_conf = joints_2d[:, :, -1] > -2.
    elif joints_2d.shape[-1] == 3:
        joints_conf = joints_2d[:, :, -1] > 0
    joints3d_conf = joints_3d[:, :, -1] != -2.

    trans = np.zeros((joints_3d.shape[0], 3), dtype=np.float32)
    if proj_mats is None:
        proj_mats = [None for _ in range(len(joints_2d))]
    if cam_dists is None:
        cam_dists = [None for _ in range(len(joints_2d))]
    # Find the translation for each example in the batch
    for i in range(joints_3d.shape[0]):
        S_i = joints_3d[i]
        joints_i = joints_2d[i, :, :2]
        valid_mask = joints_conf[i] * joints3d_conf[i]
        if valid_mask.sum() < pts_mnum:
            trans[i] = INVALID_TRANS
            continue
        if len(img_size.shape) == 1:
            imgsize = img_size
        elif len(img_size.shape) == 2:
            imgsize = img_size[i]
        else:
            raise NotImplementedError
        try:
            trans[i] = estimate_translation_cv2(S_i[valid_mask], joints_i[valid_mask],
                                                focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i],
                                                cam_dist=cam_dists[i])
        except:
            trans[i] = estimate_translation_np(S_i[valid_mask], joints_i[valid_mask],
                                               valid_mask[valid_mask].astype(np.float32),
                                               focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i])

    return torch.from_numpy(trans).float()


def batch_rodrigues(param):
    # param N x 3
    batch_size = param.shape[0]
    # 沿第二维（3个数）进行求二次范数：||x||，下面就是进行标准化，每三个数除以他们的范数。
    l1norm = torch.norm(param + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(param, angle)
    angle = angle * 0.5
    # 上面算出的是一个向量的长度：sqrt(x**2+y**2+z**2)/2,所以这个长度的的cos
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    # 用四元组表示三维旋转，有时间看一下×××××××××
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    把四元组的系数转化成旋转矩阵。四元组表示三维旋转
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def justify_detection_state(detection_flag, reorganize_idx):
    if detection_flag.sum() == 0:
        detection_flag = False
    else:
        reorganize_idx = reorganize_idx[detection_flag.bool().to(reorganize_idx.device)].long()
        detection_flag = True
    return detection_flag, reorganize_idx


def copy_state_dict(cur_state_dict, pre_state_dict, prefix='module.', drop_prefix='', fix_loaded=False):
    success_layers, failed_layers = [], []

    def _get_params(key):
        key = key.replace(drop_prefix, '')
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        if 'mano' in k:
            print(k)

        v = _get_params(k)
        try:
            if v is None:
                failed_layers.append(k)
                continue
            cur_state_dict[k].copy_(v)
            if prefix in k and prefix != '':
                k = k.split(prefix)[1]
            success_layers.append(k)
        except:
            logging.info('copy param {} failed, mismatched'.format(k))
            continue
    logging.info('missing parameters of layers:{}, {}'.format(len(failed_layers), failed_layers))
    logging.info('success layers:{}/{}, pre_state_dict have {}'.format(len(success_layers), len(cur_state_dict.keys()),
                                                                       len(pre_state_dict.keys())))
    logging.info('**************************************************************************')
    logging.info('************************** End of loading ********************************')
    logging.info('**************************************************************************')

    x = []
    for i in pre_state_dict.keys():
        if i not in success_layers:
            x.append(i)

    if fix_loaded and len(failed_layers) > 0:
        print('fixing the layers that were loaded successfully, while train the layers that failed,')
        for k in cur_state_dict.keys():
            try:
                if k in success_layers:
                    cur_state_dict[k].requires_grad = False
            except:
                print('fixing the layer {} failed'.format(k))

    return success_layers


def load_model(path, model, prefix='module.', drop_prefix='', optimizer=None, **kwargs):
    path = os.path.join(os.path.dirname(__file__), '..', path)
    logging.info('using fine_tune model: {}'.format(path))
    if os.path.exists(path):
        pretrained_model = torch.load(path, map_location='cpu')
        current_model = model.state_dict()
        if isinstance(pretrained_model, dict):
            if 'model_state_dict' in pretrained_model:
                pretrained_model = pretrained_model['model_state_dict']
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

        copy_state_dict(current_model, pretrained_model, prefix=prefix, drop_prefix=drop_prefix, **kwargs)
    else:
        logging.warning('model {} not exist!'.format(path))
        raise ValueError
    return model


def get_remove_keys(dt, keys=[]):
    targets = []
    for key in keys:
        targets.append(dt[key])
    for key in keys:
        del dt[key]
    return targets


def process_idx(reorganize_idx, vids=None):
    result_size = reorganize_idx.shape[0]
    if isinstance(reorganize_idx, torch.Tensor):
        reorganize_idx = reorganize_idx.cpu().numpy()
    used_idx = reorganize_idx[vids] if vids is not None else reorganize_idx
    used_org_inds = np.unique(used_idx)
    per_img_inds = [np.where(reorganize_idx == org_idx)[0] for org_idx in used_org_inds]

    return used_org_inds, per_img_inds


def reorganize_results(outputs, img_paths, reorganize_idx):
    results = {}
    detected = outputs['detection_flag_cache'].detach().cpu().numpy().astype(np.bool_)
    cam_results = outputs['params_dict']['cam'].detach().cpu().numpy().astype(np.float16)[detected]
    trans_results = outputs['cam_trans'].detach().cpu().numpy().astype(np.float16)[detected]
    mano_pose_results = outputs['params_dict']['poses'].detach().cpu().numpy().astype(np.float16)[detected]
    mano_shape_results = outputs['params_dict']['betas'].detach().cpu().numpy().astype(np.float16)[detected]
    joints = outputs['j3d'].detach().cpu().numpy().astype(np.float16)[detected]
    verts_results = outputs['verts'].detach().cpu().numpy().astype(np.float16)[detected]
    pj2d_results = outputs['pj2d'].detach().cpu().numpy().astype(np.float16)[detected]
    pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy().astype(np.float16)[detected]
    hand_type = outputs['output_hand_type'].detach().cpu().numpy().astype(np.int32)[detected]

    vids_org = np.unique(reorganize_idx)
    for idx, vid in enumerate(vids_org):
        verts_vids = np.where(reorganize_idx == vid)[0]
        img_path = img_paths[verts_vids[0]]
        results[img_path] = [{} for idx in range(len(verts_vids))]
        for subject_idx, batch_idx in enumerate(verts_vids):
            results[img_path][subject_idx]['cam'] = cam_results[batch_idx]
            results[img_path][subject_idx]['cam_trans'] = trans_results[batch_idx]
            results[img_path][subject_idx]['poses'] = mano_pose_results[batch_idx]
            results[img_path][subject_idx]['betas'] = mano_shape_results[batch_idx]
            results[img_path][subject_idx]['j3d'] = joints[batch_idx]
            results[img_path][subject_idx]['verts'] = verts_results[batch_idx]
            results[img_path][subject_idx]['pj2d'] = pj2d_results[batch_idx]
            results[img_path][subject_idx]['pj2d_org'] = pj2d_org_results[batch_idx]
            results[img_path][subject_idx]['hand_type'] = hand_type[batch_idx]
            results[img_path][subject_idx]['detection_flag_cache'] = detected[batch_idx]

    return results


def image_crop_pad(image, crop_trbl=(0, 0, 0, 0), bbox=None, pad_ratio=1., pad_trbl=None, draw_kp_on_image=False):
    '''
    Input args:
        image : np.array, size H x W x 3
        crop_trbl : tuple, size 4, represent the cropped size on top, right, bottom, left side, Each entry may be a single int.
        bbox : np.array/list/tuple, size 4, represent the left, top, right, bottom, we can derive the crop_trbl from the bbox
        pad_ratio : float, ratio = width / height
        pad_trbl: np.array/list/tuple, size 4, represent the pad size on top, right, bottom, left side, Each entry may be a single int.
    return:
        image: np.array, size H x W x 3
    '''
    if bbox is not None:
        assert len(bbox) == 4, print(
            'bbox input of image_crop_pad is supposed to be in length 4!, while {} is given'.format(bbox))

        def calc_crop_trbl_from_bbox(bbox, image_shape):
            l, t, r, b = bbox
            h, w = image_shape[:2]
            return (int(max(0, t)), int(max(0, w - r)), int(max(0, h - b)), int(max(0, l)))

        crop_trbl = calc_crop_trbl_from_bbox(bbox, image.shape)
    crop_func = iaa.Sequential([iaa.Crop(px=crop_trbl, keep_size=False)])
    image_aug = np.array(crop_func(image=image))
    if pad_trbl is None:
        pad_trbl = compute_paddings_to_reach_aspect_ratio(image_aug.shape, pad_ratio)
    pad_func = iaa.Sequential([iaa.Pad(px=pad_trbl, keep_size=False)])
    image_aug = pad_func(image=image_aug)

    return image_aug, None, np.array([*image_aug.shape[:2], *crop_trbl, *pad_trbl])


def image_pad_white_bg(image, pad_trbl=None, pad_ratio=1., pad_cval=255):
    if pad_trbl is None:
        pad_trbl = compute_paddings_to_reach_aspect_ratio(image.shape, pad_ratio)
    pad_func = iaa.Sequential([iaa.Pad(px=pad_trbl, keep_size=False, pad_mode='constant', pad_cval=pad_cval)])
    image_aug = pad_func(image=image)
    return image_aug, np.array([*image_aug.shape[:2], *[0, 0, 0, 0], *pad_trbl])


def process_image_ori(originImage, bbox=None):
    orgImage_white_bg, pad_trbl = image_pad_white_bg(originImage)
    image_aug, kp2ds_aug, offsets = image_crop_pad(originImage, bbox=bbox, pad_ratio=1.)
    return orgImage_white_bg, offsets


def img_preprocess(image, imgpath=None, input_size=512, single_img_input=False, bbox=None):
    # args:
    # image: bgr frame
    # image = image[:, :, ::-1]
    image_org, offsets = process_image_ori(image)

    image = torch.from_numpy(cv2.resize(image_org, (input_size, input_size), interpolation=cv2.INTER_CUBIC))
    offsets = torch.from_numpy(offsets).float()

    if single_img_input:
        image = image.unsqueeze(0).contiguous()
        offsets = offsets.unsqueeze(0).contiguous()

    input_data = {
        'image': image,
        'offsets': offsets,
        'data_set': 'internet'}  #

    if imgpath is not None:
        name = os.path.basename(imgpath)
        imgpath, name = imgpath, name
        input_data.update({'imgpath': imgpath, 'name': name})
    return input_data


class WebcamVideoStream(object):
    def __init__(self, src=0):
        if src == 'realsense':
            import pyrealsense2 as pyrs
            cfg = pyrs.config()
            cfg.enable_stream(pyrs.stream.color, 640, 480, pyrs.format.bgr8, 30)
            cfg.enable_stream(pyrs.stream.depth, 640, 480, pyrs.format.z16, 30)
            self.align = pyrs.align(pyrs.stream.color)
            self.stream = pyrs.pipeline()
            dev = self.stream.start(cfg)
            cam_int = dev.get_stream(pyrs.stream.color).as_video_stream_profile().get_intrinsics()
            self.depth_scale = dev.get_device().first_depth_sensor().get_depth_scale()
            self.cam_k = np.array([[cam_int.fx, 0, cam_int.ppx],
                                   [0, cam_int.fy, cam_int.ppy],
                                   [0, 0, 1]])
            self.frame = self.grab_realsense()
        else:
            self.stream = cv2.VideoCapture(src)
            self.frame = self.grab_webcam()

        self.src = src
        self.stopped = False

    def grab_realsense(self):
        frames = self.align.process(self.stream.wait_for_frames())
        color = frames.get_color_frame().get_data()
        depth = frames.get_depth_frame().get_data()
        if not color or not depth:
            return
        color = np.asarray(color)
        depth = np.asarray(depth) * self.depth_scale
        return color, depth

    def grab_webcam(self):
        return self.stream.read()[1], 0

    def start(self):
        Thread(target=self.update, args=()).start()

    def update(self):
        while True:
            if self.stopped:
                return
            if self.src == 'realsense':
                self.frame = self.grab_realsense()
            else:
                self.frame = self.grab_webcam()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()


def smooth_global_rot_matrix(pred_rots, OE_filter):
    rot_mat = batch_rodrigues(pred_rots[None]).squeeze(0)
    smoothed_rot_mat = OE_filter.process(rot_mat)
    smoothed_rot = rotation_matrix_to_angle_axis(smoothed_rot_mat.reshape(1, 3, 3)).reshape(-1)
    return smoothed_rot


def create_OneEuroFilter(smooth_coeff):
    return {'poses': OneEuroFilter(smooth_coeff, 0.7), 'betas': OneEuroFilter(0.6, 0.7),
            'global_orient': OneEuroFilter(smooth_coeff, 0.7)}


def smooth_results(filters, body_pose=None, body_shape=None):
    global_rot = smooth_global_rot_matrix(body_pose[:3], filters['global_orient'])
    body_pose = torch.cat([global_rot, filters['poses'].process(body_pose[3:])], 0)
    body_shape = filters['betas'].process(body_shape)
    return body_pose, body_shape


class LowPassFilter:
    def __init__(self):
        self.prev_raw_value = None
        self.prev_filtered_value = None

    def process(self, value, alpha):
        if self.prev_raw_value is None:
            s = value
        else:
            s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
        self.prev_raw_value = value
        self.prev_filtered_value = s
        return s


class OneEuroFilter:
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
        # min_cutoff: Decreasing the minimum cutoff frequency decreases slow speed jitter
        # beta: Increasing the speed coefficient(beta) decreases speed lag.
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    def compute_alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def process(self, x, print_inter=False):
        prev_x = self.x_filter.prev_raw_value
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))

        if isinstance(edx, float):
            cutoff = self.mincutoff + self.beta * np.abs(edx)
        elif isinstance(edx, np.ndarray):
            cutoff = self.mincutoff + self.beta * np.abs(edx)
        elif isinstance(edx, torch.Tensor):
            cutoff = self.mincutoff + self.beta * torch.abs(edx)
        if print_inter:
            print(self.compute_alpha(cutoff))
        return self.x_filter.process(x, self.compute_alpha(cutoff))
