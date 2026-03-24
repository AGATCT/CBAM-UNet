# CBAM-UNet
A new kind of U-Net neural network architecture, combined with CBAM attentionmechanism, achieves better performance in image segmentaion tasks like building extraction. 

Implementation framework: `PyTorch`.

## References

- CBAM attention source code: 
https://github.com/Jongchan/attention-module
- Paper: Woo S, Park J, Lee J Y, et al. CBAM: Convolutional Block Attention Module[J]. 2018. ECCV2018
https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html

## 项目简介

- `train.py`: 训练入口
- `test.py`: 测试与结果导出
- `config.py`: 统一配置文件
- `dataset.py`: 数据集读取与 DataLoader 构建
- `getmodel.py`: 模型选择入口
- `flop.py`: 使用 `torchstat` 做简单的模型复杂度统计

## 环境依赖

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `Pillow`
- `opencv-python`
- `torchstat` (`flop.py` 使用，可选)

可按自己的 CUDA / PyTorch 版本先安装好 `torch`、`torchvision`，其余依赖执行：

```bash
pip install numpy matplotlib pillow opencv-python torchstat
```

## 数据集组织
当前代码默认从 `config.py` 中的 `data_dir` 读取数据，并约定目录结构为：

```text
datasets/
|-- train/
|   |-- images/
|   `-- labels/
|-- val/
|   |-- images/
|   `-- labels/
`-- test/
    |-- images/
    `-- labels/
```

数据要求：

- 图像和标签分别放在 `images/`、`labels/` 目录下。
- 训练、验证、测试三个子集都需要分别准备。
- 图像文件名与标签文件名最好一一对应，且数量一致。
- `dataset.py` 中是分别读取 `images` 和 `labels` 目录文件列表后按索引配对。


## 运行方式
配置文件 `config.py` ：

- 数据集根目录 `data_dir`
- 训练轮数 `ne`
- 批大小 `bs`
- 学习率 `lr`
- 模型名称 `modelname`
- 类别数 `nc`


## 输出结果

训练和测试输出默认保存在：

```text
./savel_model/{modelname}_{dataname}_{loss_}_{optimizer_}_ne{ne}_bs{bs}
```

输出文件包括：

- `best_model.pt`: 验证集损失最优的模型权重
- `loss.txt`: 每个 epoch 的验证损失记录
- `loss.jpg`: 训练损失和验证损失曲线
- `result/`: 测试集预测结果图
- `result.txt`: 测试指标记录



