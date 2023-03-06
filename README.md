# 使用文档

## 环境要求

- PaddlePaddle 2.4及以上
- OS 64位操作系统
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64位版本
- pip/pip3(9.0.1+)，64位版本
- CUDA >= 10.1
- cuDNN >= 7.6

## 安装说明

### 1. 安装PaddlePaddle

```
# CUDA10.2
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```
- 更多CUDA版本或环境快速安装，请参考[PaddlePaddle快速安装文档](https://www.paddlepaddle.org.cn/install/quick)
- 更多安装方式例如conda或源码编译安装方法，请参考[PaddlePaddle安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)

请确保您的PaddlePaddle安装成功。使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle
>>> paddle.utils.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"
```
**注意**
1. 如果您希望在多卡环境下使用PaddleDetection，请首先安装NCCL

### 2. 编译自定义算子

```
# 安装依赖
git clone https://github.com/PaddlePaddle/Paddle3D.git
cd Paddle3D
pip install -r requirements.txt
pip install -e .

cd paddle3d/ops/
python setup.py install

cd ../../../

```
### 2. 安装环境

```
# 安装依赖
git clone https://github.com/wangna11BD/MonoDETR_paddle.git 

cd MonoDETR_paddle
pip install -r requirements.txt
mkdir logs
```
 
5. 下载 [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) 数据集，数据集结构如下：
    ```
    │MonoDETR_paddle/
    ├──...
    ├──data/KITTIDataset/
    │   ├──ImageSets/
    │   ├──training/
    │   ├──testing/
    ├──...
    ```
    

## 使用说明

MonoDETR模型的配置文件为`configs/monodetr.yaml`目录下


### resnet50预训练模型下载
下载预训练模型放在当前文件夹下 https://ecloud.baidu.com?t=e34a3edaf4cd7160ef09995bef241171

### 训练
```
bash train.sh configs/monodetr.yaml > logs/monodetr.log
```

### 评估
```
bash test.sh configs/monodetr.yaml
```

## 复现结果
训练环境：16G P40 cuda10.2 py3.6.8 torch1.12.0  paddle2.4
复现结果对比 		

|     | Easy | Mod. | Hard | log | 模型 |
|:--------|:-------|:-------|:-------|:---------|:---------|
| torch | 25.46% | 19.74% | 16.57% | [log](https://ecloud.baidu.com?t=e34a3edaf4cd71600a5f16968f5d2ce5) | [model](https://ecloud.baidu.com?t=e34a3edaf4cd71606478a7f0c9938556) |
| torch | 25.77% | 18.63% | 15.38% | - | - |
| paddle | 26.47% | 18.78% | 15.54% | - | - |
| paddle | 26.21% | 18.93% | 15.75% | - | - |
| paddle | 27.23% | 19.55% | 16.15%  | [log1](https://ecloud.baidu.com?t=e34a3edaf4cd71603363e3c4b41b9030)[log2](https://ecloud.baidu.com?t=e34a3edaf4cd7160609db4377c89e094) | [model](https://ecloud.baidu.com?t=e34a3edaf4cd7160c5030a592c7d9c91) |
| paddle | 23.91% | 18.10% | 15.13%| - | - |


## Citation
```bash
@article{zhang2022monodetr,
  title={MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection},
  author={Zhang, Renrui and Qiu, Han and Wang, Tai and Xu, Xuanzhuo and Guo, Ziyu and Qiao, Yu and Gao, Peng and Li, Hongsheng},
  journal={arXiv preprint arXiv:2203.13310},
  year={2022}
}
```

