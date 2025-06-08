import os, time
import numpy as np
import open3d as o3d
import torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"        
device      = torch.device("cuda")              
voxel_size  = 0.005                            

path_A = r"D:\vscodefiles\python\DensePoint\02691156\ply_file\1a54a2319e87bd4071d03b466c72ce41.ply"
path_B = r"D:\vscodefiles\python\DensePoint\02773838\ply_file\3edc3bc2ecf58aae40090e51144242ea.ply"

def voxelize(pcd: o3d.geometry.PointCloud):
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if pcd.has_colors():
        rgb = np.asarray(pcd.colors, dtype=np.float32)
    else:
        rgb = np.zeros_like(xyz)
    return xyz, rgb

def build_dense(xyz, rgb, mins_glb, dims, device):
    # 计算体素索引
    idx = np.floor((xyz - mins_glb) / voxel_size).astype(np.int64)  # (N,3)
    idx = torch.from_numpy(idx).to(device)
    rgb = torch.from_numpy(rgb).to(device)

    W, H, D = dims
    dense = torch.zeros((W, H, D, 4), dtype=torch.float32, device=device)

    # 占据
    dense[idx[:, 0], idx[:, 1], idx[:, 2], 0] = 1.0
    # 颜色 (多点落入同体素时取最后一个，可自己改成 scatter_add 求均值)
    dense[idx[:, 0], idx[:, 1], idx[:, 2], 1:] = rgb
    return dense

def reconstruct_and_show(dense_rec, mins_glb, title):
    """把 (W,H,D,4) 张量阈值化成点云并显示"""
    occ_mask   = dense_rec[..., 0] > 0.5
    coords_ijk = torch.nonzero(occ_mask).cpu().numpy()
    xyz        = coords_ijk.astype(np.float32) * voxel_size + mins_glb

    colors = dense_rec[..., 1:4][occ_mask]  # 直接索引被占用的体素颜色
    colors = colors.clip(0, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
    print(f"[INFO] {title}: {len(xyz)} points")
    o3d.visualization.draw_geometries([pcd], window_name=title)

pcd_A = o3d.io.read_point_cloud(path_A)
pcd_B = o3d.io.read_point_cloud(path_B)

xyz_A, rgb_A = voxelize(pcd_A)
xyz_B, rgb_B = voxelize(pcd_B)

mins_glb = np.minimum(xyz_A.min(0), xyz_B.min(0))
maxs_glb = np.maximum(xyz_A.max(0), xyz_B.max(0))
dims     = (np.ceil((maxs_glb - mins_glb) / voxel_size).astype(int) + 1)
W, H, D  = dims
print("Global voxel grid size:", dims)

dense_A = build_dense(xyz_A, rgb_A, mins_glb, dims, device)
dense_B = build_dense(xyz_B, rgb_B, mins_glb, dims, device)

F_A = torch.fft.fftn(dense_A, dim=(0, 1, 2))
F_B = torch.fft.fftn(dense_B, dim=(0, 1, 2))

amp_A   = torch.abs(F_A)
phase_A =torch.angle(F_A)
amp_B=torch.abs(F_B)
phase_B = torch.angle(F_B)

#-------- 混合：取 A 的振幅 + B 的相位 --------
F_mix = amp_B * torch.exp(1j * phase_A)
dense_mix = torch.fft.ifftn(F_mix, s=(W, H, D), dim=(0, 1, 2)).real
reconstruct_and_show(dense_mix, mins_glb, title="Amp(A)+Phase(B)")
print("Done.")
