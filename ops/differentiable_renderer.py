import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.raycast_rgbd.raycast_rgbd import RaycastRGBD
import pdb

class DiffRenderer(nn.Module):
    def __init__(self, cfg):
        super(DiffRenderer, self).__init__()
        self.cfg = cfg
        self.voxel_size = cfg.MODEL.VOXEL_SIZE
        self.truncation = 5
        self.ray_increment = 0.3 * self.truncation
        self.thresh_sample_dist = 5 * self.ray_increment
        self.max_num_locs_per_sample = cfg.MODEL.RERENDER.MAX_NUM_POINTS
        self.raycast_depth_max = cfg.MODEL.RERENDER.RAYCAST_DEPTH_MAX
        self.dim = cfg.MODEL.N_VOX
        
        self.bs = cfg.BATCH_SIZE
        if cfg.MODE == 'train':
            self.n_views = cfg.TRAIN.N_VIEWS
        else:
            self.n_views = cfg.TEST.N_VIEWS
        self.w = 640
        self.h = 480

        self.raycaster_rgbd = RaycastRGBD(1, self.dim, self.w, self.h, 
                                          depth_min=0.001/self.voxel_size, depth_max=self.raycast_depth_max/self.voxel_size,
                                          thresh_sample_dist=self.thresh_sample_dist, ray_increment=self.ray_increment,
                                          max_num_locs_per_sample=self.max_num_locs_per_sample, voxel_size=self.voxel_size)

    def forward(self, coords, origin, sdf, depths_target, feats, intrinsics_matrix, view_matrix):

        loss = 0
        depths = []
        for batch in range(self.bs):
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
            depths_target_batch = depths_target[batch]

            color = torch.zeros(coords_batch.shape[0], 3).cuda()    

            depths_batch = []
            coords_batch = torch.cat([coords_batch[:, 2:3], coords_batch[:, 1:2], coords_batch[:, 0:1], coords_batch[:, 3:4]], dim=1)

            for view_idx in range(self.n_views):
                normals = self.compute_normals_sparse(coords_batch, sdf_batch, self.dim, transform=view_matrix_batch[view_idx].unsqueeze(0))

                intrinsics_batch = torch.FloatTensor(1, 4).cuda()
                intrinsics_batch[:, 0] = intrinsics_matrix_batch[view_idx, 0, 0]
                intrinsics_batch[:, 1] = intrinsics_matrix_batch[view_idx, 1, 1]
                intrinsics_batch[:, 2] = intrinsics_matrix_batch[view_idx, 0, 2]
                intrinsics_batch[:, 3] = intrinsics_matrix_batch[view_idx, 1, 2]
                
                raycast_color, raycast_depth, raycast_normal = self.raycaster_rgbd(coords_batch, sdf_batch, color, normals, 
                                                                                   view_matrix_batch[view_idx].unsqueeze(0).contiguous(), 
                                                                                   intrinsics_batch, origin_batch)

                depth_target = depths_target_batch[view_idx].unsqueeze(0)

                valid = (raycast_depth != -float('inf')) & (depth_target != 0)

                if valid.sum() != 0:
                    loss += (torch.mean(torch.abs(self.normalize(raycast_depth[valid]) - self.normalize(depth_target[valid]))) * self.cfg.MODEL.RERENDER.WEIGHT) / self.n_views

                depths_batch.append(torch.clone(raycast_depth.squeeze(0)))

            depths_batch = torch.stack(depths_batch, dim=0)
            depths_batch[depths_batch == -float('inf')] = 0

            depths.append(depths_batch)
            
        depths = torch.stack(depths, dim=0)

        return loss, depths, depths_target

    def compute_normals_dense(self, sdf):
        assert(len(sdf.shape) == 5) # batch mode
        dims = sdf.shape[2:]
        sdfx = sdf[:,:,1:dims[0]-1,1:dims[1]-1,2:dims[2]] - sdf[:,:,1:dims[0]-1,1:dims[1]-1,0:dims[2]-2]
        sdfy = sdf[:,:,1:dims[0]-1,2:dims[1],1:dims[2]-1] - sdf[:,:,1:dims[0]-1,0:dims[1]-2,1:dims[2]-1]
        sdfz = sdf[:,:,2:dims[0],1:dims[1]-1,1:dims[2]-1] - sdf[:,:,0:dims[0]-2,1:dims[1]-1,1:dims[2]-1]
        return torch.cat([sdfx, sdfy, sdfz], 1)

    def compute_normals(self, sdf, sdf_locs, transform=None):
        normals = self.compute_normals_dense(sdf)
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

    def compute_normals_sparse(self, sdf_locs, sdf_vals, dims, transform=None):
        dims = [dims[0], dims[1], dims[2]]
        batch_size = int(sdf_locs[-1,-1]+1)
        sdf = torch.zeros(batch_size, 1, dims[0], dims[1], dims[2]).to(sdf_vals.device)
        sdf[sdf_locs[:,-1], :, sdf_locs[:,0], sdf_locs[:,1], sdf_locs[:,2]] = sdf_vals
        #sdf = scn.SparseToDense(3, sdf_vals.shape[1])(scn.InputLayer(3, dims, mode=0)([sdf_locs, sdf_vals]))
        normals = self.compute_normals_dense(sdf)
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

    def normalize(self, x):
        x = x - x.min()
        x = x / x.max()

        return x