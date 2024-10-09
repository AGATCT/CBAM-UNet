from config import *
from models import UNet
from models import UNet_cbam3
from torchstat import stat
model = UNet.UNet(in_channels=3, out_channels=nc)
model2 = UNet_cbam3.UNet(in_channels=3, out_channels=nc)

stat(model, (3, 512, 512))
stat(model2, (3, 512, 512))