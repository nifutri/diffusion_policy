# Save as "save_data.py"
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import torch
import dill
import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    cfg['_target_'] = 'diffusion_policy.workspace.train_diffusion_unet_robocasa_hybrid_workspace_extract_data.DataExtractionDiffusionUnetRobocasaHybridWorkspace'
    
    # Access the custom policy_type parameter
    cfg['policy']['_target_'] = 'diffusion_policy.policy.diffusion_unet_robocasa_hybrid_image_policy.DiffusionUnetRobocasaHybridImagePolicy' 
    ## Modify ckpt and logging file
    cfg['checkpoint'] = f'data/outputs/2025.06.02/11.49.39_train_diffusion_unet_robocasa_hybrid_closedrawer_image/checkpoints/latest.ckpt'
    
    cfg['logging'] = f'data/outputs/2025.06.02/11.49.39_train_diffusion_unet_robocasa_hybrid_closedrawer_image/checkpoints/latest_FD_data.pt'
    
    ## End of modification
    cfg['dataloader']['shuffle'] = False

    if 'output_file' in cfg:
        cfg['output_file'] = cfg.output_file

    OmegaConf.resolve(cfg)
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    payload = torch.load(open(cfg['checkpoint'], 'rb'), pickle_module=dill)
    print("loaded checkpoint from   ", cfg['checkpoint'])
    workspace.logging = cfg['logging']
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    workspace.run()

if __name__ == "__main__":
    main()