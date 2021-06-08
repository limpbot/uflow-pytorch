

# Requirements

## python3
sudo apt-get install python3-dev

sudo apt-get install python3.8-dev

python3 -m venv venv  
pip install pip --upgrade  

pip install torch  
pip install torchvision  
pip install opencv-python  
pip install tensorboardX  
pip install tensorboard  
pip install configargparse   
pip install kornia    
pip install matplotlib 
pip install pytorch3d

#### no longer required: 
pip install pptk  


# dev-extension
(req python3.8-dev)
pip install wandb

wandb login

cuda 10.1:
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html


# Data Structure

### calibration to kitti-flow directory:
<datasets_dir>/KITTI_flow/data_scene_flow_calib

train-dataset: \
<datasets_dir>/KITTI_flow_multiview/testing/image_2

eval-dataset: \
<datasets_dir>/KITTI_flow/training/image_2' \
<datasets_dir>/KITTI_flow/training/flow_occ' \
<datasets_dir>/KITTI_flow/training/flow_noc'

models:
.models

# Run

note: in case you want to continue to train a model use the model_tag.
e.g. '2020_08_12_v65'

python coach.py -s config/config_setup_0.yaml -c config/config_coach_def_usceneflow.yaml -e config/exps/r10/config_coach_exp_r10e0_disp_se3_mask3_val1.yaml  

python presenter.py -s config/config_setup_0.yaml -c config/config_coach_def_usceneflow.yaml -e config/present/config_presenter_0.yaml  

