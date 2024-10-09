import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from getmodel import get_model
from config import *
from metric.evaluator import Evaluator
import cv2

# 加载PyTorch模型
model = get_model(modelname)

# model.load_state_dict(torch.load(f'{savel_model_path}/last_model.pt', map_location='cuda:0'))
model.load_state_dict(torch.load(f'{savel_model_path}/best_model.pt', map_location='cuda:0'))
model.eval()

# 定义转换函数（如果需要，根据您的模型预处理要求进行调整）
transform = transforms.Compose([  # 将影像调整为模型输入大小
    transforms.ToTensor(),          # 转换为PyTorch张量
])


result_folder = f'{savel_model_path}/result/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
iou_list = []
precision_list = []
recall_list = []
f1_list = []

evaluator = Evaluator(num_class=nc)
evaluator.reset()

for image_name in os.listdir(test_img_dir):
    print(image_name)
    image_path = os.path.join(test_img_dir, image_name)

    label_path = os.path.join(test_lab_dir, image_name)  # 假设标签与影像文件名相同
    result_path = os.path.join(result_folder, image_name[:-4]+'.jpg')  # 保存预测结果的路径

    # 读取影像和标签
    image1 = cv2.imread(image_path)
    label = cv2.imread(label_path, 0)
    label[label != 0] = 1  # 将255改为1

    # 预处理影像
    input_tensor = transform(image1)
    input_tensor = input_tensor.unsqueeze(0)  # 增加批次维度

    # 使用模型进行预测
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.nn.Softmax(dim=1)(output)
        predicted_label = torch.argmax(output, dim=1)
        evaluator.add_batch(pre_image=predicted_label[0].cpu().numpy(), gt_image=label)
    # 保存预测结果
    predicted_label = predicted_label.squeeze().cpu().numpy()
    # if dataname == "whudata":
    #     predicted_image = Image.fromarray((predicted_label).astype(np.uint8))
    # elif dataname == "data":
    predicted_image = Image.fromarray((predicted_label * 255).astype(np.uint8))
    predicted_image.save(result_path)

iou_per_class = evaluator.Intersection_over_Union()
f1_per_class = evaluator.F1()
OA = evaluator.OA()
precision = evaluator.Precision()
recall = evaluator.Recall()

result_txt = f"{savel_model_path}/result.txt"


f = open(result_txt, 'a')  # 若文件不存在，系统自动创建。
for class_name, class_iou, class_f1 in zip([i for i in range(nc)], iou_per_class, f1_per_class):
    print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
    # 内容之后写入。可修改该模式（'w+','w','wb'等）
    f.write("\n")
    f.write('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))  # 将字符串写入文件中
    f.write("\n")
print('F1:{}, mIOU:{}, OA:{}, P:{}, R:{}'.format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA,
                                                 np.nanmean(precision[:-1]), np.nanmean(recall[:-1])))
print(f"{savel_model_path} --")
print('F1:{}, mIOU:{}, OA:{}, P:{}, R:{}'.format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA,
                                                 np.nanmean(precision), np.nanmean(recall)))
print('{}\n{}\n{}\n{}\n{}'.format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA,
                                                 np.nanmean(precision), np.nanmean(recall)))
# 内容之后写入。可修改该模式（'w+','w','wb'等）
f.write("\n")
f.write('F1:{}, mIOU:{}, OA:{}, P:{}, R:{}'.format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA,
                                                 np.nanmean(precision), np.nanmean(recall)))  # 将字符串写入文件中
f.write("\n")

f.write("*******************************")
f.close()