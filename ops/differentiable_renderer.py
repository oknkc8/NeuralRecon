import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

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
        
        self.raycaster_rgbd_target = RaycastRGBD(1, self.dim, self.w, self.h, 
                                                 depth_min=0.001/self.voxel_size, depth_max=self.raycast_depth_max/self.voxel_size,
                                                 thresh_sample_dist=self.thresh_sample_dist, ray_increment=self.ray_increment,
                                                 max_num_locs_per_sample=self.max_num_locs_per_sample, voxel_size=self.voxel_size)

    # def forward(self, coords, origin, sdf, sdf_target, feats, intrinsics_matrix, view_matrix):
    def forward(self, coords, origin, sdf, sdf_target, depths_target, intrinsics_matrix, view_matrix):

        depth_loss = torch.tensor(0.0).cuda()
        normal_loss = torch.tensor(0.0).cuda()
        depths = []
        #depths_target = []
        normals = []
        normals_target = []
        for batch in range(self.bs):
            coords_batch = coords[:, 1:]
            intrinsics_matrix_batch = intrinsics_matrix[batch]
            view_matrix_batch = view_matrix[batch]
            origin_batch = origin[batch].cuda()
            # origin_batch = torch.cat([origin_batch[2:3], origin_batch[1:2], origin_batch[0:1]], dim=0)
            sdf_batch = sdf
            sdf_target_batch = sdf_target
            
            num_points = coords_batch.shape[0]
            batch_index = torch.tensor([0 for _ in range(num_points)]).cuda()
            batch_index = batch_index.unsqueeze(1)
            coords_batch = torch.cat([coords_batch, batch_index], dim=1).type(torch.LongTensor).cuda()
            depths_target_batch = depths_target[batch]

            color = torch.zeros(coords_batch.shape[0], 3).cuda()    

            depths_batch = []
            # depths_target_batch = []
            normals_batch = []
            normals_target_batch = []
            coords_batch = torch.cat([coords_batch[:, 2:3], coords_batch[:, 1:2], coords_batch[:, 0:1], coords_batch[:, 3:4]], dim=1)

            for view_idx in range(self.n_views):
                normals_sparse = self.compute_normals_sparse(coords_batch, sdf_batch, self.dim, transform=view_matrix_batch[view_idx].unsqueeze(0))
                normals_target_sparse = self.compute_normals_sparse(coords_batch, sdf_target_batch, self.dim, transform=view_matrix_batch[view_idx].unsqueeze(0))

                intrinsics_batch = torch.FloatTensor(1, 4).cuda()
                intrinsics_batch[:, 0] = intrinsics_matrix_batch[view_idx, 0, 0]
                intrinsics_batch[:, 1] = intrinsics_matrix_batch[view_idx, 1, 1]
                intrinsics_batch[:, 2] = intrinsics_matrix_batch[view_idx, 0, 2]
                intrinsics_batch[:, 3] = intrinsics_matrix_batch[view_idx, 1, 2]
                
                _, raycast_depth, raycast_normal = self.raycaster_rgbd(coords_batch, sdf_batch, color, normals_sparse, 
                                                                       view_matrix_batch[view_idx].unsqueeze(0).contiguous(), 
                                                                        intrinsics_batch, origin_batch)

                _, _, raycast_normal_target = self.raycaster_rgbd_target(coords_batch, sdf_target_batch, color, normals_target_sparse, 
                                                                         view_matrix_batch[view_idx].unsqueeze(0).contiguous(), 
                                                                         intrinsics_batch, origin_batch)

                # pdb.set_trace()
                depth_target = depths_target_batch[view_idx].unsqueeze(0)

                # tmp_normal = kornia.geometry.depth_to_normals(depth_target.unsqueeze(0), intrinsics_matrix_batch[view_idx].unsqueeze(0), normalize_points=False)
                # tmp_normal = tmp_normal.permute(0,2,3,1)

                valid = (raycast_depth != -float('inf')) & (depth_target != 0)
                # valid = (raycast_depth != -float('inf')) & (raycast_depth_target != -float('inf'))

                if valid.sum() != 0:
                    depth_loss += (torch.mean(torch.abs(self.normalize(raycast_depth[valid]) - self.normalize(depth_target[valid]))) * self.cfg.MODEL.RERENDER.WEIGHT) / self.n_views
                    # loss += (torch.mean(torch.abs(self.normalize(raycast_depth[valid]) - self.normalize(raycast_depth_target[valid]))) * self.cfg.MODEL.RERENDER.WEIGHT) / self.n_views
            
                valid_normal = (raycast_normal != -float('inf')) & (raycast_normal_target != -float('inf'))

                if valid_normal.sum() != 0:
                    normal_loss += (torch.mean(torch.abs(raycast_normal[valid_normal] - raycast_normal_target[valid_normal])) * self.cfg.MODEL.RERENDER.WEIGHT) / self.n_views
                
                if torch.isnan(depth_loss):
                    pdb.set_trace()

                depths_batch.append(torch.clone(raycast_depth.squeeze(0)))
                # depths_target_batch.append(torch.clone(raycast_depth_target.squeeze(0)))
                normals_batch.append(torch.clone(raycast_normal.squeeze(0)))
                normals_target_batch.append(torch.clone(raycast_normal_target.squeeze(0)))

            depths_batch = torch.stack(depths_batch, dim=0)
            depths_batch[depths_batch == -float('inf')] = 0
            # depths_target_batch = torch.stack(depths_target_batch, dim=0)
            # depths_target_batch[depths_target_batch == -float('inf')] = 0
            normals_batch = torch.stack(normals_batch, dim=0)
            normals_batch[normals_batch == -float('inf')] = 0
            normals_target_batch = torch.stack(normals_target_batch, dim=0)
            normals_target_batch[normals_target_batch == -float('inf')] = 0
            

            depths.append(depths_batch)
            # depths_target.append(depths_target_batch)
            normals.append(normals_batch)
            normals_target.append(normals_target_batch)
            
        depths = torch.stack(depths, dim=0)
        # depths_target = torch.stack(depths_target, dim=0)
        normals = torch.stack(normals, dim=0)
        normals_target = torch.stack(normals_target, dim=0)

        return depth_loss, normal_loss, depths, depths_target, normals, normals_target

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
        
        # pdb.set_trace()

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
        if x.max() == 0:
            return x

        x = x / x.max()

        return x