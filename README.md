# CBAM-UNet
A new kind of U-Net neural network architecture, combined with CBAM attention mechanism, achieves better performance in image segmentaion tasks like building extraction. Implemented in PyTorch.

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


## 运行配置
配置文件 `config.py` ：

- 数据集根目录 `data_dir`
- 训练轮数 `ne`
- 批大小 `bs`
- 学习率 `lr`
- 模型名称 `modelname`
- 类别数 `nc`


训练和测试输出默认路径：

```text
./savel_model/{modelname}_{dataname}_{loss_}_{optimizer_}_ne{ne}_bs{bs}
```

输出文件：

- `best_model.pt`: 验证集损失最优的模型权重
- `loss.txt`: 每个 epoch 的验证损失记录
- `loss.jpg`: 训练损失和验证损失曲线
- `result/`: 测试集预测结果图
- `result.txt`: 测试指标记录



