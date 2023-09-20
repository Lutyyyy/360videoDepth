## Some implementation of Omnidirection Video Depth Estimation
### Install
```shell
git clone https://github.com/Lutyyyy/360videoDepth.git
cd 360videoDepth/
pip install -r ./dependencies/requirements.txt
```
### Train
Download the preprocessed data from [sc_depth_pl](https://github.com/JiawangBian/sc_depth_pl#dataset)
```shell
bash ./experiments/davis/train_sequence.sh 0
```
The first argument indicates the GPU id for training. <br>
For distributed parallel training, add `--multiprocess_distributed` in `./experiments/original_dataset/train_seq.sh`