# 模型参数
nc = 2 # num_class
bs = 15 # batch_size
ne = 15 # num_epochs
lr = 0.001 #学习率
num_workers = 4
loss_ = "bce" # (bce)
optimizer_ = "adam" # 优化器

# 模型配置
#数据集名称
dataname = "data"
# dataname = "whudata"
# 数据集路径
data_dir=r'./datasets'
# data_dir=r'./whudataset'

# 选择模型
# modelname = "UNet"
# modelname = "UNet_v3"
# modelname = "UNet_v4"
# modelname = "UNet_cbam"
# modelname = "UNet2_cbam"
# modelname = "UNet2"
# modelname = "UNet2_cbam2"
# modelname = "UNet2_cbam3"
# modelname = "UNet2_cbam4"
modelname = "UNet_cbam3"
# modelname = "UNet_cbam5"
# modelname = "fcn"
savel_model_path = f"./savel_model/{modelname}_{dataname}_{loss_}_{optimizer_}_ne{ne}_bs{bs}"


train_img_dir = data_dir+r"\train/images"
train_lab_dir = data_dir+r"\train/labels"

val_img_dir = data_dir+r"\val/images"
val_lab_dir = data_dir+r"\val/labels"

test_img_dir = data_dir+r"\test/images"
test_lab_dir = data_dir+r"\test/labels"



# # grlunet parameters from grl init
# i = 0                #--myself
# img_size=64
# in_channels=3
# out_channels=None
# embed_dim=96
# depths=[6, 6, 6, 6, 6, 6]
# num_heads_window=[3, 3, 3, 3, 3, 3]
# num_heads_stripe=[3, 3, 3, 3, 3, 3]
# window_size=8
# stripe_size=[8, 8]  # used for stripe window attention
# stripe_groups=[None, None]
# stripe_shift=False
# mlp_ratio=4.0
# qkv_bias=True
# qkv_proj_type="linear"
# anchor_proj_type="avgpool"
# anchor_one_stage=True
# anchor_window_down_factor=1
# out_proj_type="linear"
# local_connection=False
# drop_rate=0.0
# attn_drop_rate=0.0
# # drop_path=0.0,  #--myself
# drop_path_rate=0.1
# norm_layer=nn.LayerNorm
# pretrained_window_size=[0, 0]
# pretrained_stripe_size=[0, 0]
# conv_type="1conv"
# init_method="n"  # initialization method of the weight parameters used to train large scale models.
# # **kwargs,