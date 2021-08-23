import torch
import torch.nn.functional as F
from torch.nn.functional import grid_sample

from ops.raycast_rgbd.raycast_rgbd import RaycastRGBD
import pdb

def compute_normals_dense(sdf):
    assert(len(sdf.shape) == 5) # batch mode
    dims = sdf.shape[2:]
    sdfx = sdf[:,:,1:dims[0]-1,1:dims[1]-1,2:dims[2]] - sdf[:,:,1:dims[0]-1,1:dims[1]-1,0:dims[2]-2]
    sdfy = sdf[:,:,1:dims[0]-1,2:dims[1],1:dims[2]-1] - sdf[:,:,1:dims[0]-1,0:dims[1]-2,1:dims[2]-1]
    sdfz = sdf[:,:,2:dims[0],1:dims[1]-1,1:dims[2]-1] - sdf[:,:,0:dims[0]-2,1:dims[1]-1,1:dims[2]-1]
    return torch.cat([sdfx, sdfy, sdfz], 1)

def compute_normals(sdf, sdf_locs, transform=None):
    normals = compute_normals_dense(sdf)
    normals = torch.nn.functional.pad(normals, (1,1,1,1,1,1),value=-float('inf'))
    normals = normals[sdf_locs[:,3],:,sdf_locs[:,0],sdf_locs[:,1],sdf_locs[:,2]].contiguous()
    mask = normals[:,0] != -float('inf')
    normals[normals == -float('inf')] = 0
    if transform is not None:
        n = []
        for b in range(transform.shape[0]):
            n.append(torch.matmul(transform[b,:3,:3], normals[sdf_locs[:,-1] == b].t()).t())
        normals = torch.cat(n)
    normals = -torch.nn.functional.normalize(normals, p=2, dim=1, eps=1e-5, out=None)
    return normals

def compute_normals_sparse(sdf_locs, sdf_vals, dims, transform=None):
    dims = [dims[0], dims[1], dims[2]]
    batch_size = int(sdf_locs[-1,-1]+1)
    sdf = torch.zeros(batch_size, 1, dims[0], dims[1], dims[2]).to(sdf_vals.device)
    sdf[sdf_locs[:,-1], :, sdf_locs[:,0], sdf_locs[:,1], sdf_locs[:,2]] = sdf_vals
    #sdf = scn.SparseToDense(3, sdf_vals.shape[1])(scn.InputLayer(3, dims, mode=0)([sdf_locs, sdf_vals]))
    normals = compute_normals_dense(sdf)
    normals = torch.nn.functional.pad(normals, (1,1,1,1,1,1),value=-float('inf'))
    normals = normals[sdf_locs[:,3],:,sdf_locs[:,0],sdf_locs[:,1],sdf_locs[:,2]].contiguous()
    mask = normals[:,0] != -float('inf')
    normals[normals == -float('inf')] = 0
    if transform is not None:
        n = []
        for b in range(transform.shape[0]):
            #n.append(normals[sdf_locs[:,-1] == b])
            n.append(torch.matmul(transform[b,:3,:3], normals[sdf_locs[:,-1] == b].t()).t())
            #bmask = (sdf_locs[:,-1] == b) & mask
            #normals[bmask] = torch.matmul(transform[b,:3,:3], normals[bmask].t()).t()
        normals = torch.cat(n)
    #normals[mask] = -torch.nn.functional.normalize(normals[mask], p=2, dim=1, eps=1e-5, out=None)
    normals = -torch.nn.functional.normalize(normals, p=2, dim=1, eps=1e-5, out=None)
    return normals


def diff_renderer(cfg, coords, origin, voxel_size, sdf, depths_target, feats, intrinsics_matrix, view_matrix):
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

    truncation = 5
    ray_increment = 0.3 * truncation
    thresh_sample_dist = 5 * ray_increment
    max_num_locs_per_sample = cfg.RERENDER.MAX_NUM_POINTS
    raycast_depth_max = cfg.RERENDER.RAYCAST_DEPTH_MAX

    dim = cfg.N_VOX

    bs, n_views, h, w = depths_target.shape

    loss = 0
    depths = []
    for batch in range(bs):
        coords_batch = coords[:, 1:]
        intrinsics_matrix_batch = intrinsics_matrix[batch]
        view_matrix_batch = view_matrix[batch]
        origin_batch = origin[batch].cuda()
        # origin_batch = torch.cat([origin_batch[2:3], origin_batch[1:2], origin_batch[0:1]], dim=0)
        sdf_batch = sdf
        
        num_points = coords_batch.shape[0]
        batch_index = torch.tensor([0 for _ in range(num_points)]).cuda()
        batch_index = batch_index.unsqueeze(1)
        coords_batch = torch.cat([coords_batch, batch_index], dim=1).type(torch.LongTensor).cuda()

        raycaster_rgbd = RaycastRGBD(1, dim, w, h, depth_min=0.001/voxel_size, depth_max=raycast_depth_max/voxel_size,
                                        thresh_sample_dist=thresh_sample_dist, ray_increment=ray_increment,
                                        max_num_locs_per_sample=max_num_locs_per_sample,
                                        origin=origin_batch, voxel_size=voxel_size)
        color = torch.zeros(coords_batch.shape[0], 3).cuda()    

        depths_batch = []
        coords_batch = torch.cat([coords_batch[:, 2:3], coords_batch[:, 1:2], coords_batch[:, 0:1], coords_batch[:, 3:4]], dim=1)

        for view_idx in range(n_views):
            normals = compute_normals_sparse(coords_batch, sdf_batch, dim, transform=view_matrix_batch[view_idx].unsqueeze(0))

            intrinsics_batch = torch.FloatTensor(1, 4).cuda()
            intrinsics_batch[:, 0] = intrinsics_matrix_batch[view_idx, 0, 0]
            intrinsics_batch[:, 1] = intrinsics_matrix_batch[view_idx, 1, 1]
            intrinsics_batch[:, 2] = intrinsics_matrix_batch[view_idx, 0, 2]
            intrinsics_batch[:, 3] = intrinsics_matrix_batch[view_idx, 1, 2]
            
            raycast_color, raycast_depths, raycast_normal = raycaster_rgbd(coords_batch, sdf_batch, color, normals, 
                                                                           view_matrix_batch[view_idx].unsqueeze(0).contiguous(), intrinsics_batch)

            depths_batch.append(torch.clone(raycast_depths))
        
        depths_batch = torch.stack(depths_batch, dim=0).squeeze(1)

        depths_target_batch = depths_target[batch]

        valid = (depths_batch != -float('inf')) & (depths_target_batch != 0)

        if valid.sum() != 0:
            loss += (torch.mean(torch.abs(normalize(depths_batch[valid]) - normalize(depths_target_batch[valid]))) * cfg.RERENDER.WEIGHT)

        depths_batch[depths_batch == -float('inf')] = 0

        depths.append(depths_batch)
    
    
    depths = torch.stack(depths, dim=0)

    return loss, depths, depths_target

def normalize(x):
    x = x - x.min()
    x = x / x.max()

    return x