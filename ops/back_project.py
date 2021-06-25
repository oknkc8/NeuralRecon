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

    print('-'*10 + 'back_project' + '-'*10)
    print('n_views:', n_views)
    print('bs:', bs)
    print('c:', c)
    print('h:', h)
    print('w:', w)

    feature_volume_all = torch.zeros(coords.shape[0], c + 1).cuda()
    count = torch.zeros(coords.shape[0]).cuda()
    print('feature_volume_all:', feature_volume_all.shape)
    print('count:', count.shape)

    for batch in range(bs):
        print('\n'+'=======================')
        print('batch:', batch)
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        coords_batch = coords[batch_ind][:, 1:]

        print('batch_ind:', batch_ind.shape)
        print('coords_batch:', coords_batch.shape)
        print('-'*20)

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]
        proj_batch = KRcam[:, batch]

        print('coords_batch:', coords_batch.shape)
        print('origin_batch:', origin_batch.shape)
        print('feats_batch:', feats_batch.shape)
        print('proj_batch:', proj_batch.shape)
        print('-'*20)

        grid_batch = coords_batch * voxel_size + origin_batch.float()
        print('grid_batch:', grid_batch.shape)
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        print('rs_grid:', rs_grid.shape)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        print('rs_grid:', rs_grid.shape)
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)      # for homogeneous coordinate
        print('rs_grid:', rs_grid.shape)
        print('-'*20)

        # Project grid
        im_p = proj_batch @ rs_grid
        print('im_p:', im_p.shape)
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z
        print('-'*20)

        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        print('im_grid:', im_grid.shape)
        mask = im_grid.abs() <= 1
        print('mask:', mask.shape)
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)
        print('mask:', mask.shape)
        print('-'*20)

        feats_batch = feats_batch.view(n_views, c, h, w)
        print('feats_batch:', feats_batch.shape)
        im_grid = im_grid.view(n_views, 1, -1, 2)
        print('im_grid:', im_grid.shape)
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)
        print('features:', features.shape)
        print('-'*20)

        features = features.view(n_views, c, -1)
        print('features:', features.shape)
        mask = mask.view(n_views, -1)
        print('mask:', mask.shape)
        im_z = im_z.view(n_views, -1)
        print('im_z:', im_z.shape)
        print('-'*20)
        # remove nan
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0

        count[batch_ind] = mask.sum(dim=0).float()
        print('count:', count.shape)
        print('-'*20)

        # aggregate multi view
        features = features.sum(dim=0)
        print('features:', features.shape)
        mask = mask.sum(dim=0)
        print('mask:', mask.shape)
        invalid_mask = mask == 0
        mask[invalid_mask] = 1
        in_scope_mask = mask.unsqueeze(0)
        print('in_scope_mask:', in_scope_mask.shape)
        features /= in_scope_mask
        features = features.permute(1, 0).contiguous()
        print('features:', features.shape)

        # concat normalized depth value
        im_z = im_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()
        print('im_z:', im_z.shape)
        im_z_mean = im_z[im_z > 0].mean()
        im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
        im_z_norm = (im_z - im_z_mean) / im_z_std
        im_z_norm[im_z <= 0] = 0
        features = torch.cat([features, im_z_norm], dim=1)
        print('features:', features.shape)

        feature_volume_all[batch_ind] = features
    return feature_volume_all, count
