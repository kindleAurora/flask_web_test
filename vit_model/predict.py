
import cv2
import torch
from PIL import Image
from torchvision import transforms

import torch.nn.functional as F
import numpy as np
from vit_model import vit_large_patch16_224_in21k as create_model



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = r"F:\picture data deep learning\W23\CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209\CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209\Train\living\000048\000001.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img,(224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    # [N, C, H, W]
    img = data_transform(img_pil)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model = create_model(num_classes=2, has_logits=True).to(device)
    # load model weights
    model_weight_path = r"flask_web_test-day26\model-11.pth"
    # 加载保存的数据
    saved_data = torch.load(model_weight_path)

    # 恢复模型状态字典
    model.load_state_dict(saved_data['model_state_dict'], strict=False)

    model.eval()
    with torch.no_grad():
        # predict class
        output= model(img.to(device))
        print(output)
        
        x_cha= torch.max(output,dim=1)[0].item()-torch.min(output,dim=1)[0].item()
        sscore= 1-np.exp(-x_cha)
        print(sscore)
        
        pred_classes = torch.max(output, dim=1)[1]
        print(pred_classes)
        element = pred_classes.item()
        print(element)
         
        

if __name__ == '__main__':
    
    main()
