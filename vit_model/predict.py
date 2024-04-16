import os
import cv2
import torch
import numpy as np
import math
from PIL import Image

from torchvision import transforms

import torch.nn.functional as F

from vit_model import vit_large_patch16_224_in21k as create_model


class Detection:
    def __init__(self):
        caffemodel = "weight\Widerface-RetinaFace.caffemodel"
        deploy = "weight\deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3] * width, out[max_conf_index, 4] * height, \
                                   out[max_conf_index, 5] * width, out[max_conf_index, 6] * height
        bbox = [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]
        return bbox



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = Detection()
    data_transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = r"F:\picture data deep learning\W23\CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209\CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209\Train\living\000002\000001.jpg"
    
    img = cv2.imread(img_path)
    image_bbox = detector.get_bbox(img)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image)
    # [N, C, H, W]
    image = data_transform(img_pil)
    # expand batch dimension
    image = torch.unsqueeze(image, dim=0)

    model = create_model(num_classes=2, has_logits=True).to(device)
    # load model weights
    model_weight_path = "model-11.pth"
    # 加载保存的数据
    saved_data = torch.load(model_weight_path)

    # 恢复模型状态字典
    model.load_state_dict(saved_data['model_state_dict'], strict=False)


    model.eval()
    with torch.no_grad():
        # predict class
        output= model(image.to(device))
        pred_classes = torch.max(output, dim=1)[1]

        aaa = torch.max(output, dim = 1)[0].item() - torch.min(output, dim = 1)[0].item()
        value = 1 - np.exp(-aaa)
        
        print(value) 
        if pred_classes == 1:
            result_text = "RealFace Score: {:.3f}".format(value)
            color = (255, 0, 0)
        else:
            result_text = "FakeFace Score: {:.3f}".format(value)
            color = (0, 0, 255)
        cv2.rectangle(
            img,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            img,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*img.shape[0]/1024, color)
    cv2.imshow("Detected Objects", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
