# Installation

This document contains detailed instructions for installing dependencies for RPT. We recommand using the [install.sh](install.sh). The code is tested on an Ubuntu 16.04 system with Nvidia GPU (We recommand 1080TI / TITAN XP).

### Requirments
* Conda with Python 3.6.
* Nvidia GPU.
* PyTorch 1.1.0
* yacs
* pyyaml
* matplotlib
* tqdm
* OpenCV

## Step-by-step instructions

#### Create environment and activate
```bash
conda create --name siamreppoints python=3.6
conda activate siamreppoints
```

#### Install numpy/pytorch/opencv
```
conda install numpy
conda install pytorch=1.1.0 torchvision cuda90 -c pytorch
pip install opencv-python
```

#### Install other requirements
```
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX
```

#### Build extensions
```
python setup.py build_ext --inplace
```

### Build extensions of DCN
```
python ./siamreppoints/setup.py build_ext --inplace
```

## Try with scripts
```
bash install.sh /path/to/your/conda siamreppoints
```
