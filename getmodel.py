from config import *
from models import UNet
# from models import UNet_v2
from models import UNet_v3
from models import UNet_v4
from models import UNet_cbam
from models import UNet2_cbam
from models import UNet2
from models import UNet2_cbam2
from models import UNet2_cbam3
from models import UNet2_cbam4
from models import UNet_cbam3
from models import UNet_cbam5
from models import fcn
def get_model(model_name):
    if model_name == "UNet":
        model_ = UNet.UNet(in_channels=3, out_channels=nc)
    # elif model_name == "UNet_v2":
    #     model_ = UNet_v2.UNet(in_channels=3, out_channels=nc)
    elif model_name == "UNet_v3":
        model_ = UNet_v3.UNet(in_channels=3, out_channels=nc)
    elif model_name == "UNet_v4":
        model_ = UNet_v4.UNet(in_channels=3, out_channels=nc)
    elif model_name == "UNet_cbam":
        model_ = UNet_cbam.UNet(in_channels=3, out_channels=nc)
    elif model_name == "UNet2_cbam":
        model_ = UNet2_cbam.UNet(in_channels=3, out_channels=nc)
    elif model_name == "UNet2":
        model_ = UNet2.UNet(in_channels=3, out_channels=nc)
    elif model_name == "UNet2_cbam2":
        model_ = UNet2_cbam2.UNet(in_channels=3, out_channels=nc)
    elif model_name == "UNet2_cbam3":
        model_ = UNet2_cbam3.UNet(in_channels=3, out_channels=nc)
    elif model_name == "UNet2_cbam4":
        model_ = UNet2_cbam4.UNet(in_channels=3, out_channels=nc)
    elif model_name == "UNet_cbam3":
        model_ = UNet_cbam3.UNet(in_channels=3, out_channels=nc)
    elif model_name == "UNet_cbam5":
        model_ = UNet_cbam5.UNet(in_channels=3, out_channels=nc)
    elif model_name == "fcn":
        model_ = fcn.FCN(num_classes=nc)

    return model_