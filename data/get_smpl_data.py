import numpy as np
import smplx
import torch

from smplx.lbs import lbs


def main():
    smpl_kwargs = dict(model_type="smpl", dtype=torch.float32, create_transl=True, gender="male")
    model = smplx.create(model_path="SMPLX_NEUTRAL.npz", **smpl_kwargs)
    torch.manual_seed(0)
    
    betas = torch.rand(1, 10)
    thetas = torch.rand(1, 66)
    t = torch.rand(1, 3)
    model_output = model.forward(
        betas=betas,
        body_pose=thetas[..., 3:],
        global_orient=thetas[..., :3],
        transl=t,
        pose2rot=True
    )

    np.savez("test/smpl_forward.npz", 
        betas=betas.detach().numpy(),
        thetas=thetas.detach().numpy(),
        t=t.detach().numpy(),
        joints=model_output.joints[0].detach().numpy(),
        vertices=model_output.vertices[0].detach().numpy()
    )


if __name__ == "__main__":
    main()
