import torch
import torch.nn as nn
import torchvision
import Pre_Process
vgg_16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
vgg_16.classifier[6] = nn.Linear(4096, 1)
vgg_16.load_state_dict(torch.load('./Facial_Beauty_Detect_V01.pt')['model_state_dict'])
vgg_16.eval()
while(True):
    path = input('Input path:')
    img=Pre_Process.test_one_img(path)
    print("Score:",vgg_16(img).item())
