"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from termcolor import colored

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    # cls = hydra.utils.get_class(cfg._target_)
    # workspace: BaseWorkspace = cls(cfg)
    # # print(colored(f"Running experiment {rollout_idx+1}/50", "green"))
    # workspace.run_all()


    for rollout_idx in range(0, 50):
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        print(colored(f"Running experiment {rollout_idx+1}/50", "green"))
        workspace.run_single_idx(rollout_idx)
        # workspace.run()

if __name__ == "__main__":
    main()
