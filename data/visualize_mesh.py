import smplx
import torch
import vedo


betas = torch.rand(1, 10)
pose = torch.zeros(1, 72)
t = torch.rand(1, 3)

smpl = smplx.SMPL("/Users/sele/models/smpl/smpl_pkl/SMPL_MALE.pkl")
smpl_output = smpl.forward(betas, body_pose=pose[:, 3:], global_orient=pose[:, :3], transl=t)
vertices = smpl_output.vertices.detach().numpy()[0]
faces = smpl.faces

mesh = vedo.Mesh([vertices, faces])
mesh.show()
