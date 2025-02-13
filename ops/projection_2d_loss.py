import torch
import torch.nn.functional as F
from torch.nn.functional import grid_sample


def projection_2d_loss(cfg, coords, origin, voxel_size, tsdf, depth_target, feats, KRcam):
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

    bs, n_views, h, w = depth_target.shape
    _, _, _, h_feat, w_feat = feats.shape

    loss = 0
    depths = []
    depths_target = []
    depths_target_masked = []
    for batch in range(bs):
        # print('\t=======================')
        # print('\tbatch:', batch)
        #batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        #coords_batch = coords[batch_ind][:, 1:]
        coords_batch = coords[:, 1:]

        # print('\tcoords_batch:', coords_batch.shape)
        # print('\t'+ '-'*20)

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        #feats_batch = feats[:, batch]
        depth_target_batch = depth_target[batch]
        #print('KRcam:', KRcam)
        proj_batch = KRcam[:, batch]
        #print('proj_batch:', proj_batch)

        # print('\tcoords_batch:', coords_batch.shape)
        # print('\torigin_batch:', origin_batch.shape)
        #print('\tproj_batch:', proj_batch.shape)
        # print('\t'+ '-'*20)

        grid_batch = coords_batch * voxel_size + origin_batch.float()
        # print('\tgrid_batch:', grid_batch.shape)
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        # print('\trs_grid:', rs_grid.shape)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        # print('\trs_grid:', rs_grid.shape)
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)      # for homogeneous coordinate
        # print('\trs_grid:', rs_grid.shape)
        # print('\t'+ '-'*20)

        # Project grid
        im_p = proj_batch @ rs_grid
        # print('\tim_p:', im_p.shape)
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]

        # print('im_x:', im_x.shape)
        # print('im_x unique:', im_x.unique())
        # print('im_y:', im_y.shape)
        # print('im_y unique:', im_y.unique())
        # print('im_z:', im_z.shape)
        # print('im_z unique:', im_z.unique())


        im_x = im_x / im_z      # normalize x, y
        im_y = im_y / im_z
        # print('\t'+ '-'*20)

        im_grid = torch.stack([2 * im_x / (w_feat - 1) - 1, 2 * im_y / (h_feat - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        # print('\tmask:', mask.shape)
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)     # mask.sum(dim=-1) == 2 : both im_x and im_y is positive and less than 1 
                                                        # im_z > 0 : ahead of normal plane (im_z=0)
        # print('\tmask:', mask.shape)
        # print('\t'+ '-'*20)

        im_x = im_grid[:, :, 0]
        im_y = im_grid[:, :, 1]
        # print('im_x:', im_x.shape)
        # print('im_y:', im_y.shape)

        im_grid = im_grid.view(n_views, 1, -1, 2)
        # print('\tim_grid:', im_grid.shape)
        
        # print('\t'+ '-'*20)

        mask = mask.view(n_views, -1)
        # print('\tmask:', mask.shape)
        im_z = im_z.view(n_views, -1)
        im_z[mask == False] = 0

        # print('im_x:', im_x.grad_fn)        


        im_x = (im_x / 2 + 0.5) * w_feat
        im_y = (im_y / 2 + 0.5) * h_feat

        mask[im_x.round() >= w_feat] = False
        mask[im_y.round() >= h_feat] = False

        
        tsdf_mask = tsdf.abs() <= cfg.PROJECTION.THRESHOLD
        # tmp = torch.unique(tsdf.view(-1))
        # print('tsdf:', torch.sort(tsdf.view(-1)).values)
        #print('tsdf:', torch.unique(tsdf.view(-1)))
        tsdf_mask = tsdf_mask.permute(1, 0)
        tsdf_mask = tsdf_mask.expand(n_views, -1)
        #print('tsdf_mask:', tsdf_mask.grad_fn)
        # print('tsdf_mask:', tsdf_mask.sum())


        """
            connect gradient flow for im_z(depth) and tsdf value
        """
        diff_im_z = tsdf.permute(1,0)
        diff_im_z = diff_im_z.expand(n_views, -1).clone()
        diff_im_z[diff_im_z.abs() <= cfg.PROJECTION.THRESHOLD] = 1
        diff_im_z[diff_im_z != 1] = 0

        im_z = diff_im_z * im_z

        new_mask = mask & tsdf_mask

        tsdf = tsdf.squeeze()
        upscale_depth = []
        for view_idx in range(n_views):
            view_new_mask = new_mask[view_idx]
            view_im_x = im_x[view_idx]
            view_im_y = im_y[view_idx]
            view_im_z = im_z[view_idx]

            view_im_x = view_im_x[view_new_mask]
            view_im_y = view_im_y[view_new_mask]
            view_im_z = view_im_z[view_new_mask]

            view_upscale_depth = torch.ones([h_feat * cfg.PROJECTION.UPSCALE + 1, w_feat * cfg.PROJECTION.UPSCALE + 1]).cuda()
            # print('view_upscale_depth:', view_upscale_depth.shape)
            view_upscale_depth *= 100            

            # print('max im x:', view_im_x.max())
            # print('min im x:', view_im_x.min())
            # print('max im y:', view_im_y.max())
            # print('min im y:', view_im_y.min())
            # print('view_im_x:', view_im_x.unique())
            # print('view_im_y:', view_im_y.unique())
            view_im_x = (view_im_x * cfg.PROJECTION.UPSCALE).round().long()
            view_im_y = (view_im_y * cfg.PROJECTION.UPSCALE).round().long()

            # print('view_im_x:', view_im_x.unique())
            # print('view_im_y:', view_im_y.unique())

            view_im_x = view_im_x.view(-1)
            view_im_y = view_im_y.view(-1)
            view_im_z = view_im_z.view(-1)

            view_upscale_depth[view_im_y, view_im_x] = view_im_z
            

            upscale_depth.append(view_upscale_depth)

        upscale_depth = torch.stack(upscale_depth, dim=0)

        # im_z 미리 normalize 필요!
        # 필요하면 depth_target도 미리 normalize 필요

        
        
        downscale_depth = -1 * F.adaptive_max_pool2d(-1 * upscale_depth, output_size = (h_feat, w_feat))
        downscale_depth[downscale_depth == 100] = 0

        downscale_depth = downscale_depth.unsqueeze(1)
        depth_target_batch = depth_target_batch.unsqueeze(1)
        # print('downscale_depth:',downscale_depth.shape)
        # print('depth_target_batch:',depth_target_batch.shape)

        #depth_target_batch = F.interpolate(depth_target_batch, size=[h_feat, w_feat])

        downscale_depth = F.interpolate(downscale_depth, size=[h, w])
        # print('depth_target_batch:',depth_target_batch.shape)

        downscale_depth = downscale_depth.squeeze(1)
        depth_target_batch = depth_target_batch.squeeze(1)
        # print('downscale_depth:',downscale_depth.shape)
        # print('depth_target_batch:',depth_target_batch.shape)
        """
        downscale_depth = upscale_depth
        downscale_depth[downscale_depth == 100] = 0
        print('downscale_dpeth:',downscale_depth.grad_fn)
        print('=====')
        """

        #target_mask = (downscale_depth != 0)    # only regressed depth
        #target_mask = (depth_target_batch != 0)     # only gt depth
        target_mask = (downscale_depth != 0) & (depth_target_batch != 0) # both

        depths_target_masked_batch = depth_target_batch * target_mask
        downscale_depth_masked = downscale_depth * target_mask

        downscale_depth_masked = F.normalize(downscale_depth_masked, dim=0)
        depths_target_masked_batch = F.normalize(depths_target_masked_batch, dim=0)

        #loss += (F.l1_loss(downscale_depth, depths_target_masked_batch) * cfg.PROJECTION.LOSS_WEIGHT)
        loss += (F.l1_loss(downscale_depth_masked, depths_target_masked_batch) * cfg.PROJECTION.LOSS_WEIGHT)

        depths.append(downscale_depth)
        depths_target.append(depth_target_batch)
        depths_target_masked.append(depths_target_masked_batch)

    depths = torch.stack(depths, dim=0)
    depths_target = torch.stack(depths_target, dim=0)
    depths_target_masked = torch.stack(depths_target_masked, dim=0)
    return loss, depths, depths_target, depths_target_masked
