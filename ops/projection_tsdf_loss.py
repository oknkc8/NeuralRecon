import torch
import torch.nn.functional as F
from torch.nn.functional import grid_sample

from utils import sparse_to_dense_channel, sparse_to_dense_torch, apply_log_transform


def fov_tsdf_loss(cfg, coords, origin, voxel_size, tsdf, tsdf_target, grid_mask, KRcam, feats):
    '''
    Unproject the image fetures to form a 3D (sparse) feature volume

    :param coords: coordinates of voxels,
    dim: (num of voxels, 4) (4 : batch ind, x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (num of voxels, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (num of voxels,)
    '''


    n_views, bs, _, h, w = feats.shape
    m = cfg.FOV_TSDF_LOSS.DEPTH_INDEX_MAX

    loss = 0
    grid_mask = grid_mask.float().detach()

    for batch in range(bs):
        coords_batch = coords[:, 1:].view(-1, 3)
        tsdf_batch = tsdf[:, batch]
        tsdf_target_batch = tsdf_target[:, batch]
        grid_mask_batch = grid_mask
        proj_batch = KRcam[:, batch]
        proj_batch_inv = proj_batch.inverse()
        origin_batch = origin[batch].unsqueeze(0)

        # print('grid_mask_batch:', grid_mask_batch.unique())

        # change tsdf sparse to dense
        tsdf_dense_batch = sparse_to_dense_torch(coords_batch, tsdf_batch, cfg.N_VOX, 0, tsdf_batch.device)
        tsdf_target_dense_batch = sparse_to_dense_torch(coords_batch, tsdf_target_batch, cfg.N_VOX, 0, tsdf_target_batch.device)
        grid_mask_dense_batch = sparse_to_dense_torch(coords_batch, grid_mask_batch, cfg.N_VOX, 0, grid_mask_batch.device).cuda()

        # print('grid_mask_dense_batch:', grid_mask_dense_batch.unique())

        # generate grid for camera coordinate
        grid_cam_dim = [m, h, w]
        grid_cam_range = [torch.arange(-1, 1, 2 / grid_cam_dim[axis]) for axis in range(3)]
        grid_cam = torch.stack(torch.meshgrid(grid_cam_range[0], grid_cam_range[1], grid_cam_range[2]), dim=-1)
        grid_cam = torch.stack([grid_cam for _ in range(n_views)]).cuda()

        grid_cam = grid_cam.view(n_views, -1, 3)
        grid_cam = grid_cam.permute(0, 2, 1).contiguous()
        nV = grid_cam.shape[-1]

        # change to ndc
        im_x = grid_cam[:, 0, :].unsqueeze(1)
        im_y = grid_cam[:, 1, :].unsqueeze(1)
        im_z = grid_cam[:, 2, :].unsqueeze(1)

        im_x = im_x * (w - 1) / 2
        im_y = im_y * (h - 1) / 2

        grid_cam = torch.cat([im_x, im_y, im_z, torch.ones([n_views, 1, nV]).cuda()], dim=1)

        # transform to world coordinate
        grid_world = proj_batch_inv @ grid_cam
        grid_world = grid_world.permute(0, 2, 1)[:, :, :3]
        grid_world = (grid_world - origin_batch) / voxel_size
        grid_world = grid_world.reshape(n_views, m, w, h, 3)

        # normalize grid_world -> [-1, 1]
        
        for axis in range(3):
            grid_world[:,:,:,:, axis] /= cfg.N_VOX[axis]
            grid_world[:,:,:,:, axis] -= 0.5

        grid_world_valid = (grid_world[:,:,:,:, 0].abs()<=1)* (grid_world[:,:,:,:, 1].abs()<=1)* (grid_world[:,:,:,:, 2].abs()<=1)
        grid_world_valid = grid_world_valid.float()
        
        for view_idx in range(n_views):
            # print(tsdf_dense_batch.unsqueeze(0).unsqueeze(0).shape)
            # print(grid_world[view_idx].unsqueeze(0).shape)
            sampled_tsdf_dense = F.grid_sample(tsdf_dense_batch.unsqueeze(0).unsqueeze(0), grid_world[view_idx].unsqueeze(0), align_corners=True)
            sampled_tsdf_target_dense = F.grid_sample(tsdf_target_dense_batch.unsqueeze(0).unsqueeze(0), grid_world[view_idx].unsqueeze(0), align_corners=True)
            sampled_grid_mask_dense = F.grid_sample(grid_mask_dense_batch.unsqueeze(0).unsqueeze(0), grid_world[view_idx].unsqueeze(0), align_corners=True)
            #print(grid_world)
            #print('sampled_grid_mask_dense:', sampled_grid_mask_dense.max())

            if view_idx == 0:
                sampled_tsdf_dense_batch = sampled_tsdf_dense
                sampled_tsdf_target_dense_batch = sampled_tsdf_target_dense
                sampled_grid_mask_dense_batch = sampled_grid_mask_dense
            else:
                sampled_tsdf_dense_batch = torch.cat([sampled_tsdf_dense_batch, sampled_tsdf_dense], dim=0)
                sampled_tsdf_target_dense_batch = torch.cat([sampled_tsdf_target_dense_batch, sampled_tsdf_target_dense], dim=0)
                sampled_grid_mask_dense_batch = torch.cat([sampled_grid_mask_dense_batch, sampled_grid_mask_dense], dim=0)
        
        sampled_tsdf_dense_batch = sampled_tsdf_dense_batch.squeeze(1)
        sampled_tsdf_target_dense_batch = sampled_tsdf_target_dense_batch.squeeze(1)
        sampled_grid_mask_dense_batch = sampled_grid_mask_dense_batch.squeeze(1)

        sampled_tsdf_dense_batch *= grid_world_valid
        sampled_tsdf_target_dense_batch *= grid_world_valid

        # print(sampled_tsdf_dense_batch.max())
        # print(sampled_tsdf_target_dense_batch.max())
        # print(sampled_grid_mask_dense_batch.max())

        sampled_grid_mask_dense_batch = sampled_grid_mask_dense_batch != 0

        if sampled_grid_mask_dense_batch.max() == 0:
            return 0.0

        sampled_tsdf_dense_batch = sampled_tsdf_dense_batch[sampled_grid_mask_dense_batch]
        sampled_tsdf_target_dense_batch = sampled_tsdf_target_dense_batch[sampled_grid_mask_dense_batch]

        sampled_tsdf_dense_batch = apply_log_transform(sampled_tsdf_dense_batch)
        sampled_tsdf_target_dense_batch = apply_log_transform(sampled_tsdf_target_dense_batch)

        loss += torch.mean(torch.abs(sampled_tsdf_dense_batch - sampled_tsdf_target_dense_batch))
        # print(loss)

    return loss * cfg.FOV_TSDF_LOSS.WEIGHT
