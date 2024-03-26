from flask import Flask, request, Response, render_template, send_from_directory,url_for,redirect,jsonify
import cv2
import os
import sys
import base64
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

import warnings
import time
import random

import torch
from torchvision import transforms
from vit_model.vit_model import vit_large_patch16_224_in21k as create_model


from flask_socketio import SocketIO
from io import BytesIO
from PIL import Image
from engineio.payload import Payload
warnings.filterwarnings('ignore')

# 初始化model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
model = create_model(num_classes=2, has_logits=True).to(device)
# load model weights
model_weight_path = r"flask_web_test-day26\model-11.pth"
# 加载保存的数据
saved_data = torch.load(model_weight_path)
# 恢复模型状态字典
model.load_state_dict(saved_data['model_state_dict'], strict=False)
model.eval()




# 定义flask应用app入口
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'byd flask sm dou buhui'

# 使用socket来进行通信交互
Payload.max_decode_packets = 50000
socketio = SocketIO(cors_allowed_origins='*',ping_timeout=60, ping_interval=20)
socketio.init_app(app)

@socketio.on('connect')
def handle_connect():
    print('客户端已连接')

@socketio.on('message')
def handle_message(message):
    print('收到消息:', message)
    # 在这里处理接收到的消息，并可以向客户端发送消息

@socketio.on('disconnect')
def handle_disconnect():
    print('客户端已断开连接')



# 收到需要检测摄像头图片
@socketio.on('videoFrame_need_checked')
def videoFrame(data):
    
    # 使用 socket 返回结果给前端
    # 从 Data URL 中提取出 Base64 编码的图像数据部分
    img_data = data.split(',')[1]
    # 解码 Base64 编码的图像数据
    img_binary = base64.b64decode(img_data)

    # 将二进制数据转换为PIL图像对象
    img = Image.open(BytesIO(img_binary))

    # 将图像对象转换为 NumPy 数组
    img_array = np.asarray(img)
    # 图像格式标准化
    img = cv2.resize(img_array,(224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    # [N, C, H, W]
    img = data_transform(img_pil)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    
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
         
    
    print('ok,video')
    socketio.emit('detection_result_withoutpicture', {'confidence': sscore, 'classification': element})



    
# 转到不同页面
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/404')
def a404():
    return render_template('404.html')

@app.route('/antifraud')
def antifraud():
    return render_template('antifraud.html')

@app.route('/facegame')
def facegame():
    return render_template('facegame.html')

@app.route('/faceinput')
def faceinput():
    return render_template('faceinput.html')


    
    

# 停止摄像头
@app.route('/stop_camera',methods=['GET','POST'])
def stop_camera():
    global camera
    if camera is not None:
        camera.release()  # 释放摄像头资源
        camera = None
        return 'Camera released successfully'
    else:
        return 'Camera is not in use'

    
# 图片检测
@app.route('/detect_picture', methods=['POST'])
def detect_picture():
    # 获取上传的图片文件
    file = request.files['image']
    
    # 读取图片数据
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 复制原始图像
    img_copy = img.copy()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    # [N, C, H, W]
    img = data_transform(img_pil)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
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
    
    # 这里假设直接返回原图的 base64 编码
    _, img_encoded = cv2.imencode('.jpg', img_copy)
    img_base64 = base64.b64encode(img_encoded)
    img_url = 'data:image/jpeg;base64,' + img_base64.decode('utf-8')
    print('ok,picture')
    # 返回检测结果给前端
    socketio.emit('detection_result_picture', {'confidence': sscore, 'classification': element,'url':img_url})
    return 'picture ok'
    
#关于游戏界面的全都在下面(写的依托，草了)
@app.route('/game')
def game():
    return render_template('ytgame.html')

# 定义照片文件夹路径
photo_folder = '9'
#photos = [os.path.join(photo_folder,file)for file in os.listdir(photo_folder)if file.endswith(('.jpg','.png','.jpeg'))]
 
# 随机发送图片
@app.route('/randomm')
def randomm():
    
    random_photo = random.choice(photos)
    with open(random_photo,"rb") as image_file:
        image_content = image_file.read()  # 读取文件内容
        img_base64 = base64.b64encode(image_content)
        img_url = 'data:image/jpeg;base64,' + img_base64.decode('utf-8')
        # TODO
        imge_class = 0
    # 发送字节流到web
    socketio.emit('photo_data',{'picture':img_url,'class':imge_class})
    return "Random picture sent to the web"

def run():
    app.run(host='0.0.0.0', port="5000", threaded=True, debug=True)

    
if __name__ == '__main__':
    # app.run(debug=True,host='0.0.0.0',port="8080")
    socketio.run(app)
