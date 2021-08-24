import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse.tensor import PointTensor
from loguru import logger

from models.modules import SPVCNN
from utils import apply_log_transform
from .gru_fusion import GRUFusion
from ops.back_project import back_project
from ops.generate_grids import generate_grid
from ops.projection_2d_loss import projection_2d_loss
from ops.projection_tsdf_loss import fov_tsdf_loss
#from ops.differentiable_renderer import diff_renderer
from ops.differentiable_renderer import DiffRenderer


class NeuConNet(nn.Module):
    '''
    Coarse-to-fine network.
    '''

    def __init__(self, cfg):
        super(NeuConNet, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.MODEL.THRESHOLDS) - 1

        alpha = int(self.cfg.MODEL.BACKBONE2D.ARC.split('-')[-1])
        ch_in = [80 * alpha + 1, 96 + 40 * alpha + 2 + 1, 48 + 24 * alpha + 2 + 1, 24 + 24 + 2 + 1]
        channels = [96, 48, 24]

        if self.cfg.MODEL.FUSION.FUSION_ON:
            # GRU Fusion
            self.gru_fusion = GRUFusion(cfg.MODEL, channels)
        # sparse conv
        self.sp_convs = nn.ModuleList()
        # MLPs that predict tsdf and occupancy.
        self.tsdf_preds = nn.ModuleList()
        self.occ_preds = nn.ModuleList()
        for i in range(len(cfg.MODEL.THRESHOLDS)):
            self.sp_convs.append(
                SPVCNN(num_classes=1, in_channels=ch_in[i],
                       pres=1,
                       cr=1 / 2 ** i,
                       vres=self.cfg.MODEL.VOXEL_SIZE * 2 ** (self.n_scales - i),
                       dropout=self.cfg.MODEL.SPARSEREG.DROPOUT)
            )
            self.tsdf_preds.append(nn.Linear(channels[i], 1))
            # self.tsdf_preds.append(nn.Sequential(
            #                         nn.Linear(channels[i], 1),
            #                         nn.Tanh()
            #                     ))
            self.occ_preds.append(nn.Linear(channels[i], 1))
            # self.occ_preds.append(nn.Sequential(
            #                         nn.Linear(channels[i], 1),
            #                         nn.Sigmoid()
            #                     ))
        
        self.raycaster = DiffRenderer(cfg)
        

    def get_target(self, coords, inputs, scale):
        '''
        Won't be used when 'fusion_on' flag is turned on
        :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        :param inputs: (List), inputs['tsdf_list' / 'occ_list']: ground truth volume list, [(B, DIM_X, DIM_Y, DIM_Z)]
        :param scale:
        :return: tsdf_target: (Tensor), tsdf ground truth for each predicted voxels, (N,)
        :return: occ_target: (Tensor), occupancy ground truth for each predicted voxels, (N,)
        '''
        with torch.no_grad():
            tsdf_target = inputs['tsdf_list'][scale]
            occ_target = inputs['occ_list'][scale]
            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            tsdf_target = tsdf_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            return tsdf_target, occ_target

    def upsample(self, pre_feat, pre_coords, interval, num=8):
        '''

        :param pre_feat: (Tensor), features from last level, (N, C)
        :param pre_coords: (Tensor), coordinates from last level, (N, 4) (4 : Batch ind, x, y, z)
        :param interval: interval of voxels, interval = scale ** 2
        :param num: 1 -> 8
        :return: up_feat : (Tensor), upsampled features, (N*8, C)
        :return: up_coords: (N*8, 4), upsampled coordinates, (4 : Batch ind, x, y, z)
        '''
        # print('\t'+'='*10 + 'upsample' + '='*10)
        with torch.no_grad():
            pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]
            n, c = pre_feat.shape
            
            # print('\tpre_feat:', pre_feat.shape)
            # print('\tpre_coords:', pre_coords.shape)
            # print('\t=======================')

            up_feat = pre_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()

            # print('\tup_feat:', up_feat.shape)
            # print('\tup_coords:', up_coords.shape)
            # print('\t=======================')

            for i in range(num - 1):
                up_coords[:, i + 1, pos_list[i]] += interval

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 4)

            # print('\tup_feat:', up_feat.shape)
            # print('\tup_coords:', up_coords.shape)
            # print('\t=======================')

        return up_feat, up_coords

    def forward(self, features, inputs, outputs, apply_loss=False):
        '''

        :param features: list: features for each image: eg. list[0] : pyramid features for image0 : [(B, C0, H, W), (B, C1, H/2, W/2), (B, C2, H/2, W/2)]
        :param inputs: meta data from dataloader
        :param outputs: {}
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
        }
        :return: loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
        }
        '''
        bs = features[0][0].shape[0]
        pre_feat = None
        pre_coords = None
        loss_dict = {}
        loss_dict[f'projection_loss'] = 0.0
        loss_dict[f'fov_tsdf_loss'] = 0.0
        loss_dict[f'rerender_loss'] = 0.0
        image_dict = {}
        """ ----coarse to fine---- """
        for i in range(self.cfg.MODEL.N_LAYER):
            interval = 2 ** (self.n_scales - i)
            scale = self.n_scales - i
            # print('\n' + '='*80)
            # print('Interval & Scale:', interval, scale)

            if i == 0:
                # print('-'*10 + 'Generate New coords' + '-'*10)
                # """ ----generate new coords---- """
                coords = generate_grid(self.cfg.MODEL.N_VOX, interval)[0]
                # print('coords:', coords.shape)
                up_coords = []
                for b in range(bs):
                    up_coords.append(torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
                up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
                # print('up_coords:', up_coords.shape)
            else:
                # print('-'*10 + 'Upsample Coords' + '-'*10)
                # """ ----upsample coords---- """
                up_feat, up_coords = self.upsample(pre_feat, pre_coords, interval)
                # print('up_feat:', up_feat.shape)
                # print('up_coords:', up_coords.shape)


            """ ----back project---- """
            # print('-'*10 + 'Back Project' + '-'*10)
            
            # print('features:', len(features))
            # for j, feature in enumerate(features[0]):
            #     print('%dth feature:'%(j), feature.shape)

            # depths = inputs['depth']
            # print(type(inputs['depth']))
            # for j, depth in enumerate(depths):
            #     print('%dth depth:'%(j), depth.shape)

            feats = torch.stack([feat[scale] for feat in features])
            KRcam = inputs['proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous()
            # print('feats:', feats.shape)
            # print('KRcam:', KRcam.shape)
            volume, count = back_project(up_coords, inputs['vol_origin_partial'], self.cfg.MODEL.VOXEL_SIZE, feats,
                                         KRcam)
            grid_mask = count > 1

            """ ----concat feature from last stage---- """
            # print('-'*10 + 'concat feature from last stage' + '-'*10)
            if i != 0:
                feat = torch.cat([volume, up_feat], dim=1)
            else:
                feat = volume

            if not self.cfg.MODEL.FUSION.FUSION_ON:
                tsdf_target, occ_target = self.get_target(up_coords, inputs, scale)

            """ ----convert to aligned camera coordinate---- """
            # print('-'*10 + 'convert to aligned camera coordinate' + '-'*10)
            r_coords = up_coords.detach().clone().float()
            for b in range(bs):
                batch_ind = torch.nonzero(up_coords[:, 0] == b).squeeze(1)
                coords_batch = up_coords[batch_ind][:, 1:].float()
                coords_batch = coords_batch * self.cfg.MODEL.VOXEL_SIZE + inputs['vol_origin_partial'][b].float()
                coords_batch = torch.cat((coords_batch, torch.ones_like(coords_batch[:, :1])), dim=1)
                coords_batch = coords_batch @ inputs['world_to_aligned_camera'][b, :3, :].permute(1, 0).contiguous()
                r_coords[batch_ind, 1:] = coords_batch

            # batch index is in the last position
            # print('r_coords:', r_coords.shape)
            r_coords = r_coords[:, [1, 2, 3, 0]]
            # print('r_coords:', r_coords.shape)
            # print(r_coords)

            """ ----sparse conv 3d backbone---- """
            # print('-'*10 + 'sparse conv 3d backbone' + '-'*10)
            point_feat = PointTensor(feat, r_coords)
            # print('point_feat.F:', point_feat.F.shape)
            # print('point_feat.C:', point_feat.C.shape)
            feat = self.sp_convs[i](point_feat)
            # print('feat:',feat.shape)

            """ ----gru fusion---- """
            # print('-'*10 + 'gru fusion' + '-'*10)
            if self.cfg.MODEL.FUSION.FUSION_ON:
                up_coords, feat, tsdf_target, occ_target = self.gru_fusion(up_coords, feat, inputs, i)
                # print('up_coords:', up_coords.shape)
                # print('feat:', feat.shape)
                # print('tsdf_target:', tsdf_target.shape)
                # print('occ_target:', occ_target.shape)
                if self.cfg.MODEL.FUSION.FULL:
                    grid_mask = torch.ones_like(feat[:, 0]).bool()
                    # print('grid_mask:', grid_mask.shape)

            tsdf = self.tsdf_preds[i](feat)
            occ = self.occ_preds[i](feat)
            # print('tsdf:', tsdf.shape)
            # print('occ:', occ.shape)

            """ -------compute loss------- """
            depths_gt = inputs['depth']
            depth_KRcam = inputs['depth_proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous()
            intrinsics = inputs['intrinsics']
            extrinsics = inputs['extrinsics']

            # print('tsdf_target:', tsdf_target.shape)                
            # print('tsdf_target:', tsdf_target.unique())
            # print('tsdf:', tsdf.detach().unique())
            if tsdf_target is not None:
                loss = self.compute_loss(tsdf, occ, tsdf_target, occ_target,
                                         mask=grid_mask,
                                         pos_weight=self.cfg.MODEL.POS_WEIGHT)
                
                if apply_loss and i == self.cfg.MODEL.N_LAYER - 1:
                    if self.cfg.MODEL.RERENDER.LOSS:
                        rerender_loss, depths, depths_target = self.raycaster(up_coords, inputs['vol_origin_partial'], tsdf, 
                                                                              depths_gt, feats, intrinsics, extrinsics)
                        loss += rerender_loss
                        loss_dict[f'rerender_loss'] += rerender_loss    
                        image_dict.update({f'depth': depths})
                        image_dict.update({f'depth_target': depths_target})

                    if self.cfg.MODEL.PROJECTION.LOSS:
                        projection_loss, depths, depths_target, depths_target_masked = projection_2d_loss(self.cfg.MODEL, up_coords, inputs['vol_origin_partial'],
                                                                                                        self.cfg.MODEL.VOXEL_SIZE, tsdf, depths_gt, feats, KRcam)
                        loss += projection_loss

                        loss_dict[f'projection_loss'] += projection_loss
                        image_dict.update({f'projection_depth_{i}': depths})
                        image_dict.update({f'projection_depth_target_masked{i}': depths_target_masked})
                        image_dict.update({f'projection_depth_target': depths_target})

                    # if self.cfg.MODEL.FOV_TSDF_LOSS.LOSS:
                    #     fov_loss = fov_tsdf_loss(self.cfg.MODEL, up_coords, inputs['vol_origin_partial'],
                    #                              self.cfg.MODEL.VOXEL_SIZE, tsdf, tsdf_target, grid_mask, KRcam, feats)
                        
                    #     loss += fov_loss
                    #     loss_dict[f'fov_tsdf_loss'] += fov_loss
            else:
                loss = torch.Tensor(np.array([0]))[0]
            loss_dict.update({f'tsdf_occ_loss_{i}': loss})


            """ ------define the sparsity for the next stage----- """
            # print('-'*10 + 'define the sparsity for the next stage' + '-'*10)
            occupancy = occ.squeeze(1) > self.cfg.MODEL.THRESHOLDS[i]
            occupancy[grid_mask == False] = False
            # print('occupancy:', occupancy.shape)

            num = int(occupancy.sum().data.cpu())
            # print('num:', num)

            if num == 0:
                logger.warning('no valid points: scale {}'.format(i))
                return outputs, loss_dict, image_dict

            """ ------avoid out of memory: sample points if num of points is too large----- """
            # print('-'*10 + 'avoid out of memory: sample points if num of points is too large' + '-'*10)
            # print('up_coords:', up_coords.shape)
            # print('num:', num)
            # print('num_preserve:', num - self.cfg.MODEL.TRAIN_NUM_SAMPLE[i] * bs)
            if self.training and num > self.cfg.MODEL.TRAIN_NUM_SAMPLE[i] * bs:
                choice = np.random.choice(num, num - self.cfg.MODEL.TRAIN_NUM_SAMPLE[i] * bs,
                                          replace=False)
                ind = torch.nonzero(occupancy)
                occupancy[ind[choice]] = False
                # print('choice:', choice.shape)

            pre_coords = up_coords[occupancy]
            # print('pre_coords:', pre_coords.shape)
            for b in range(bs):
                batch_ind = torch.nonzero(pre_coords[:, 0] == b).squeeze(1)
                if len(batch_ind) == 0:
                    logger.warning('no valid points: scale {}, batch {}'.format(i, b))
                    return outputs, loss_dict, image_dict

            pre_feat = feat[occupancy]
            pre_tsdf = tsdf[occupancy]
            pre_occ = occ[occupancy]

            # print('-'*20)
            # print(i)
            # print('pre_feat:', pre_feat.shape)
            # print('pre_tsdf:', pre_tsdf.shape)
            # print('pre_occ:', pre_occ.shape)

            pre_feat = torch.cat([pre_feat, pre_tsdf, pre_occ], dim=1)
            # print('pre_feat:', pre_feat.shape)

            if i == self.cfg.MODEL.N_LAYER - 1:
                # print('-'*20)
                outputs['coords'] = pre_coords
                outputs['tsdf'] = pre_tsdf
                #outputs['tsdf'] = tsdf_target[occ_target.squeeze(1) > self.cfg.MODEL.THRESHOLDS[i]]
                # print('pre_tsdf:', pre_tsdf.shape)
                # print('tsdf_target:', tsdf_target.shape)
                # print('tsdf:', tsdf.shape)
                # print('occupancy:', occupancy.shape)
                # print('occ:', occ.shape)
                # print('occ_target:', occ_target.shape)

        return outputs, loss_dict, image_dict

    @staticmethod
    def compute_loss(tsdf, occ, tsdf_target, occ_target, loss_weight=(1, 1),
                     mask=None, pos_weight=1.0):
        '''

        :param tsdf: (Tensor), predicted tsdf, (N, 1)
        :param occ: (Tensor), predicted occupancy, (N, 1)
        :param tsdf_target: (Tensor),ground truth tsdf, (N, 1)
        :param occ_target: (Tensor), ground truth occupancy, (N, 1)
        :param loss_weight: (Tuple)
        :param mask: (Tensor), mask voxels which cannot be seen by all views
        :param pos_weight: (float)
        :return: loss: (Tensor)
        '''
        # compute occupancy/tsdf loss
        tsdf = tsdf.view(-1)
        occ = occ.view(-1)
        tsdf_target = tsdf_target.view(-1)
        occ_target = occ_target.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            tsdf = tsdf[mask]
            occ = occ[mask]
            tsdf_target = tsdf_target[mask]
            occ_target = occ_target[mask]

        n_all = occ_target.shape[0]
        n_p = occ_target.sum()
        if n_p == 0:
            logger.warning('target: no valid voxel when computing loss')
            return torch.Tensor([0.0]).cuda()[0] * tsdf.sum()
        w_for_1 = (n_all - n_p).float() / n_p
        w_for_1 *= pos_weight

        # compute occ bce loss
        occ_loss = F.binary_cross_entropy_with_logits(occ, occ_target.float(), pos_weight=w_for_1)

        # compute tsdf l1 loss
        tsdf = apply_log_transform(tsdf[occ_target])
        tsdf_target = apply_log_transform(tsdf_target[occ_target])
        tsdf_loss = torch.mean(torch.abs(tsdf - tsdf_target))

        # compute final loss
        loss = loss_weight[0] * occ_loss + loss_weight[1] * tsdf_loss
        return loss
