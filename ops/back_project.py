import torch
from torch.nn.functional import grid_sample


def back_project(coords, origin, voxel_size, feats, KRcam):
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
    n_views, bs, c, h, w = feats.shape

    # print()
    # print('\t'+'='*10 + 'back_project' + '='*10)
    # print('\tn_views:', n_views)
    # print('\tbs:', bs)
    # print('\tc:', c)
    # print('\th:', h)
    # print('\tw:', w)

    feature_volume_all = torch.zeros(coords.shape[0], c + 1).cuda()
    count = torch.zeros(coords.shape[0]).cuda()
    # print('\tfeature_volume_all:', feature_volume_all.shape)
    # print('\tcount:', count.shape)

    for batch in range(bs):
        # print('\t=======================')
        # print('\tbatch:', batch)
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        coords_batch = coords[batch_ind][:, 1:]

        # print('\tbatch_ind:', batch_ind.shape)
        # print('\tcoords_batch:', coords_batch.shape)
        # print('\t'+ '-'*20)

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]
        proj_batch = KRcam[:, batch]

        # print('\tcoords_batch:', coords_batch.shape)
        # print('\torigin_batch:', origin_batch.shape)
        # print('\tfeats_batch:', feats_batch.shape)
        # print('\tproj_batch:', proj_batch.shape)
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
        # print('coords_batch', coords_batch)
        # print('rs_grid:', rs_grid)
        # print('proj_batch:', proj_batch)

        im_p = proj_batch @ rs_grid
        # print('\tim_p:', im_p.shape)
        
        # print('im_p:', im_p)

        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z      # normalize x, y
        im_y = im_y / im_z
        # print('\t'+ '-'*20)

        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        # print('\tim_grid:', im_grid.shape)
        # print('im_grid:', im_grid)
        mask = im_grid.abs() <= 1
        # print('\tmask:', mask.shape)
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)     # mask.sum(dim=-1) == 2 : both im_x and im_y is positive and less than 1 
                                                        # im_z > 0 : ahead of normal plane (im_z=0)
        #print('\tmask:', mask.grad_fn)
        # print('\t'+ '-'*20)

        feats_batch = feats_batch.view(n_views, c, h, w)
        # print('\tfeats_batch:', feats_batch.shape)
        im_grid = im_grid.view(n_views, 1, -1, 2)
        # print('\tim_grid:', im_grid.shape)
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)
        # print('\tfeatures:', features.shape)
        # print('\t'+ '-'*20)

        features = features.view(n_views, c, -1)
        # print('\tfeatures:', features.shape)
        mask = mask.view(n_views, -1)
        # print('\tmask:', mask.shape)
        im_z = im_z.view(n_views, -1)
        # print('\tim_z:', im_z.shape)
        # print('\t'+ '-'*20)
        # remove nan
        # print('features:', features.is_leaf)
        # print('features:', features.grad_fn)
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        # print('features:', features.is_leaf)
        # print('features:', features.grad_fn)
        im_z[mask == False] = 0

        count[batch_ind] = mask.sum(dim=0).float()
        # print('\tcount:', count.grad_fn)
        # print('\t'+ '-'*20)

        # aggregate multi view
        features = features.sum(dim=0)
        # print('\tfeatures:', features.shape)
        mask = mask.sum(dim=0)
        # print('\tmask:', mask.shape)
        invalid_mask = mask == 0
        mask[invalid_mask] = 1              # turn false to true because of preventing divide by zero
                                            # "features /= in_scope_mask"
        in_scope_mask = mask.unsqueeze(0)
        # print('\tin_scope_mask:', in_scope_mask.shape)
        features /= in_scope_mask           # voting
        features = features.permute(1, 0).contiguous()
        # print('\tfeatures:', features.shape)
        # print('\t'+ '-'*20)

        # concat normalized depth value
        im_z = im_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()
        # print('\tim_z:', im_z.shape)
        im_z_mean = im_z[im_z > 0].mean()
        im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
        im_z_norm = (im_z - im_z_mean) / im_z_std
        im_z_norm[im_z <= 0] = 0
        features = torch.cat([features, im_z_norm], dim=1)
        # print('\tfeatures:', features.shape)
        # print('\t'+ '-'*20)

        feature_volume_all[batch_ind] = features
    
    # print()
    return feature_volume_all, count
