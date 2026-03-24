# CBAM-UNet
A new kind of U-Net neural network architecture, combined with CBAM attentionmechanism, achieves better performance in image segmentaion tasks like building extraction.

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

## 代码组织
项目包含以下部分：

```text
CBAM-UNet/
|-- config.py
|-- dataset.py
|-- flop.py
|-- getmodel.py
|-- README.md
|-- test.py
|-- train.py
|-- UNet_cbam.py
|-- datasets/
|   |-- train/
|   |   |-- images/
|   |   `-- labels/
|   |-- val/
|   |   |-- images/
|   |   `-- labels/
|   `-- test/
|       |-- images/
|       `-- labels/
|-- savel_model/
|-- models/
`-- metric/
```



## 数据集组织方式
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
这个项目没有命令行参数解析，默认通过直接修改 `config.py` 来控制实验。

- 数据集根目录 `data_dir`
- 训练轮数 `ne`
- 批大小 `bs`
- 学习率 `lr`
- 模型名称 `modelname`
- 类别数 `nc`


## 配置文件
`config.py` 是整个项目最重要的配置入口。下面按当前代码行为说明每个参数的作用。

| 参数 | 当前默认值 | 作用 |
| --- | --- | --- |
| `nc` | `2` | 分割类别数，同时影响模型输出通道数和测试评估器类别数。当前代码更适合二分类分割。 |
| `bs` | `15` | `DataLoader` 的 batch size。 |
| `ne` | `15` | 训练 epoch 数。 |
| `lr` | `0.001` | 学习率。 |
| `num_workers` | `4` | `DataLoader` 使用的 worker 数。Windows 下如果有多进程读取问题，可以尝试改成 `0`。 |
| `loss_` | `"bce"` | 当前主要用于拼接保存目录名，并没有真正切换损失函数。 |
| `optimizer_` | `"adam"` | 当前主要用于拼接保存目录名，并没有真正切换优化器。 |
| `dataname` | `"data"` | 数据集名称标记，主要用于保存目录命名。 |
| `data_dir` | `./datasets` | 数据集根目录。 |
| `modelname` | `"UNet_cbam3"` | 选择训练/测试使用的模型。 |
| `savel_model_path` | 自动拼接 | 模型权重、日志和结果的保存目录。 |
| `train_img_dir` | 自动拼接 | 训练图像目录。 |
| `train_lab_dir` | 自动拼接 | 训练标签目录。 |
| `val_img_dir` | 自动拼接 | 验证图像目录。 |
| `val_lab_dir` | 自动拼接 | 验证标签目录。 |
| `test_img_dir` | 自动拼接 | 测试图像目录。 |
| `test_lab_dir` | 自动拼接 | 测试标签目录。 |


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



