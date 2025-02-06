### ENV

```shell
conda create -n pytorch python=3.13
conda activate pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install notebook
pip install opencv-python
pip install tqdm
pip install matplotlib
pip install albumentations
pip install tensorboard
pip3 install nvitop
pip install torch-tb-profiler
pip install standard-imghdr
```

### RUN

```shell
conda activate pytorch
python format_busi.py
python compute_mean_std.py
python train.py
python infer.py
```
