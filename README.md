## PyTorch implementation of "LF-AFnet: Angular-Flexible Network for Light Field Image Super-Resolution". 

## Requirements
* **Python 3.5+**
* **PyTorch 1.0+(http://pytorch.org/)**
* **Skimage**
* **Numpy**
* **MatLab (For training/test data generation)**

## Datasets
We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets for both training and test. Please first download our dataset via [Baidu Drive](https://pan.baidu.com/s/144kg-c94EIJrzSkd-wxK9A) (key:nudt), and place the 5 datasets to the folder `./datasets/`.

## Generate Data for Training/Test
* **Run `Generate_Data_for_train.m` to generate training data. The generated data will be saved in `./data_for_train/` (x2_SR, x4_SR).**
* **Run `Generate_Data_for_test.m` to generate input LFs of the test set. The generated data will be saved in `./data_for_test/` (x2_SR, x4_SR).**


## Train and Test as our paper
* **Run `train.py` to perform network training and validation. Note that, the training settings in `train.py` should match the generated training data.**
* **Checkpoint will be saved to `./log/`.**
* **During the training, validation patches will be printed in `./log/`.**

We also provide the solution of distributed GPUs. Technically there are no limits to the operation system to run the code, but Linux system is recommended, on which the project has been tested.
```
$ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --model_name LF_AFnet --scale_factor 2 --data_name ALL --batch_size 2 --use_pre_ckpt True --path_pre_pth ./LF_AFnet_x2_epoch_50_model.pth --path_demo ./demo/ --angRes_demo 5 --path_demo_result ./demo_result/ 
```
```
$ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --model_name LF_AFnet --scale_factor 4 --data_name ALL --batch_size 2 --use_pre_ckpt True --path_pre_pth ./LF_AFnet_x4_epoch_50_model.pth --path_demo ./demo/ --angRes_demo 5 --path_demo_result ./demo_result/ 
```


## Test on your own LFs
We provide our pretrained models. You can run `demo.py` to easily perform network inference, and your LFs saved in `./demo/` will be super-resolved. The output will be saved in`./demo_result/`. Note that, the angular resolution (i.e., angRes) of your LFs is required.
```
$ python demo.py --model_name LF_AFnet --scale_factor 2 --use_pre_ckpt True  --path_pre_pth ./LF_AFnet_x2_epoch_50_model.pth --path_demo ./demo/ --angRes_demo 5 --path_demo_result ./demo_result/ 
```
```
$ python demo.py --model_name LF_AFnet --scale_factor 4 --use_pre_ckpt True --path_pre_pth ./LF_AFnet_x4_epoch_50_model.pth --path_demo ./demo/ --angRes_demo 5 --path_demo_result ./demo_result/ 
```


## Citiation
**If you find this work helpful, please consider citing the following paper:**

## Acknowledgement
**The imresize part of our code is referred from [matlab_imresize](https://github.com/fatheral/matlab_imresize). We thank the author for sharing codes.**


## Contact
**Any question regarding this work can be addressed to zyliang@nudt.edu.cn and wangyingqian16@nudt.edu.cn.**



