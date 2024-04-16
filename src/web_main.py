from flask import Flask, request, Response, render_template, send_from_directory,url_for,redirect,jsonify
import cv2
import cv2 as cv
import os
import sys
import base64
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

import warnings
import time
import random
# 人脸反欺诈
import torch
from torchvision import transforms
from vit_model.vit_model import vit_large_patch16_224_in21k as create_model
import math
# 年龄性别
import time
import argparse #argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。argparse模块的作用是用于解析命令行参数。

from flask_socketio import SocketIO
from io import BytesIO
from PIL import Image
from engineio.payload import Payload
warnings.filterwarnings('ignore')

# 初始化model
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

# 人脸反欺诈检测模型加载
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detector = Detection()
data_transform = transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
model = create_model(num_classes=2, has_logits=True).to(device)
# load model weights
model_weight_path = r"model-11.pth"
# 加载保存的数据
saved_data = torch.load(model_weight_path)
# 恢复模型状态字典
model.load_state_dict(saved_data['model_state_dict'], strict=False)
model.eval()

# 年龄性别检测模型加载

#置信度阈值conf_threshold=0.7
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    #将blob放入神经网络。计算输入的前向传递，将结果存储为 detections
    net.setInput(blob)
    ##net.forward()是个四维的返回值，标签、置信度、目标位置的4个坐标信息[xmin ymin xmax ymax]
    detections = net.forward()
    #bboxes 存储检测出的人脸
    bboxes = []
    #detections.shape[2] 可以得到检测结果的数量
    for i in range(detections.shape[2]):
        ##提取与数据相关的置信度（即概率）#预测，给出了第i个盒子预测的置信度
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            #
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument("--device", default="gpu", help="Device to inference on")

args = parser.parse_args()


#args = parser.parse_args()

faceProto = "AgeGender-main/opencv_face_detector.pbtxt"
faceModel = "AgeGender-main/opencv_face_detector_uint8.pb"

ageProto = "AgeGender-main/age_deploy.prototxt"
ageModel = "AgeGender-main/age_net.caffemodel"

genderProto = "AgeGender-main/gender_deploy.prototxt"
genderModel = "AgeGender-main/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)


if args.device == "cpu":
    ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

    genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    
    faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

    print("Using CPU device")
elif args.device == "gpu":
    ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")



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


# 使用摄像头
@app.route('/video_capture')
def video_capture():
    global camera
    camera = cv2.VideoCapture(0)
    
    while True:
        start_time = time.time()  # 记录开始时间
        frame_count = 0  # 重置帧数计数器
        success, frame = camera.read()  # read the camera frame
        if not success:
            print('sorry,not success')
            break
        image_bbox = detector.get_bbox(frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)
        # [N, C, H, W]
        image = data_transform(img_pil)
        # expand batch dimension
        image = torch.unsqueeze(image, dim=0)
        
        with torch.no_grad():
            # predict class
            output= model(image.to(device))
            pred_classes = torch.max(output, dim=1)[1]

            aaa = torch.max(output, dim = 1)[0].item() - torch.min(output, dim = 1)[0].item()
            value = 1 - np.exp(-aaa)
            if pred_classes == 1:
                result_text = "RealFace Score: {:.3f}".format(value)
                color = (255, 0, 0)
            else:
                result_text = "FakeFace Score: {:.3f}".format(value)
                color = (0, 0, 255)
            cv2.rectangle(
                frame,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                color, 2)
            cv2.putText(
                frame,
                result_text,
                (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 1.5*frame.shape[0]/1024, color)
                
        # 计算帧率
        elapsed_time = time.time() - start_time  # 计算时间间隔
        frame_count += 1  # 帧数加1
        fps = frame_count / elapsed_time  # 计算帧率
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode()  # 将字节对象转换为字符串
        img_url = 'data:image/jpeg;base64,' + img_base64
        print(value)
        print(pred_classes)
       
        # 提取pred_classes中的值并转换为Python标量
        
        pred_class_value = pred_classes.item()
        # 使用 socket 返回结果给前端
        socketio.emit('detection_result', {'confidence': value, 'classification': pred_class_value, 'url': img_url})
    return 'nihao,摄像头'

# 本地检测年龄
@app.route('/age_detection_by_camera')
def age_detection_by_camera():
    global camera
    camera = cv2.VideoCapture(0)
    padding = 20
    while True:
        start_time = time.time()  # 记录开始时间
        frame_count = 0  # 重置帧数计数器
        success, frame = camera.read()  # read the camera frame
        if not success:
            print('sorry,not success')
            break
        
        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue

        for bbox in bboxes:
        # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            #np.argmax()是用于取得数组中每一行或者每一列的的最大值。常用于机器学习中获取分类结果、计算精确度等。
            #np.max:返回数组中的最大值；  np.argmax: 返回数组中最大值坐标
            gender = genderList[genderPreds[0].argmax()]
            # print("Gender Output : {}".format(genderPreds))

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            # 去除括号
            age = age.replace("(", "").replace(")", "")

            label = "{} years old".format(age)
            cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
            
        # 计算帧率
        elapsed_time = time.time() - start_time  # 计算时间间隔
        frame_count += 1  # 帧数加1
        fps = frame_count / elapsed_time  # 计算帧率
        cv2.putText(frameFace, "FPS: {:.2f}".format(fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', frameFace)
        img_base64 = base64.b64encode(buffer).decode()  # 将字节对象转换为字符串
        img_url = 'data:image/jpeg;base64,' + img_base64
        # 使用 socket 返回结果给前端
        socketio.emit('detection_result', {'confidence': '111', 'classification': '222', 'url': img_url})
    return 'nihao,摄像头'

# 本地检测性别
@app.route('/gender_detection_by_camera')
def gender_detection_by_camera():
    global camera
    camera = cv2.VideoCapture(0)
    padding = 20
    while True:
        start_time = time.time()  # 记录开始时间
        frame_count = 0  # 重置帧数计数器
        success, frame = camera.read()  # read the camera frame
        if not success:
            print('sorry,not success')
            break
        
        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue

        for bbox in bboxes:
        # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            #np.argmax()是用于取得数组中每一行或者每一列的的最大值。常用于机器学习中获取分类结果、计算精确度等。
            #np.max:返回数组中的最大值；  np.argmax: 返回数组中最大值坐标
            gender = genderList[genderPreds[0].argmax()]
            # print("Gender Output : {}".format(genderPreds))

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            # 去除括号
            age = age.replace("(", "").replace(")", "")

            label = "{} ".format(gender)
            cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
            
        # 计算帧率
        elapsed_time = time.time() - start_time  # 计算时间间隔
        frame_count += 1  # 帧数加1
        fps = frame_count / elapsed_time  # 计算帧率
        cv2.putText(frameFace, "FPS: {:.2f}".format(fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', frameFace)
        img_base64 = base64.b64encode(buffer).decode()  # 将字节对象转换为字符串
        img_url = 'data:image/jpeg;base64,' + img_base64
        # 使用 socket 返回结果给前端
        socketio.emit('detection_result', {'confidence': '111', 'classification': '222', 'url': img_url})
    return 'nihao,摄像头'        
        
        
        
        
                      
        
# 转到不同页面
@app.route('/')
def index():
    return render_template('index.html')

# 404界面
@app.route('/404')
def a404():
    return render_template('404.html')

# 反欺诈检测
@app.route('/online_detection')
def online_detection():
    return render_template('online_detection.html')

# 年龄检测
@app.route('/age_detection')
def age_detection():
    return render_template('age_detection.html')

# 性别检测
@app.route('/gender_detection')
def gender_detection():
    return render_template('gender_detection.html')

# 人脸反欺诈检测
@app.route('/fas_detection')
def fas_detection():
    return render_template('fas_detection.html')

# 本来想做的游戏界面
@app.route('/facegame')
def facegame():
    return render_template('facegame.html')
# 技术路线
@app.route('/technology')
def technology():
    return render_template('technology.html')
# 团队成员介绍
@app.route('/resume')
def resume():
    return render_template('resume.html')

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
    
# 停止视频检测
@app.route('/stop_video',methods=['POST','GET'])
def stop_video():
    print('stop')
    if cap is not None:
        cap.release()  # 释放视频资源
        
        return 'video released successfully'
    else:
        return 'video is not in use'
    
# 图片检测人脸反欺诈
@app.route('/detect_picture', methods=['POST'])
def detect_picture():
    # 获取上传的图片文件
    file = request.files['image']
    
    # 读取图片数据
    nparr = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    image_bbox = detector.get_bbox(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    # [N, C, H, W]
    img = data_transform(img_pil)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    with torch.no_grad():
        # predict class
        output= model(img.to(device))
        pred_classes = torch.max(output, dim=1)[1]
        
        aaa = torch.max(output, dim = 1)[0].item() - torch.min(output, dim = 1)[0].item()
        value = 1 - np.exp(-aaa)
        if pred_classes == 1:
                result_text = "RealFace Score: {:.3f}".format(value)
                color = (255, 0, 0)
        else:
            result_text = "FakeFace Score: {:.3f}".format(value)
            color = (0, 0, 255)
        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 1.5*image.shape[0]/1024, color)

    # 这里假设直接返回原图的 base64 编码
    _, img_encoded = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(img_encoded)
    img_url = 'data:image/jpeg;base64,' + img_base64.decode('utf-8')
    print('ok,picture')
    # 提取pred_classes中的值并转换为Python标量
    print(value)
    print(pred_classes)    
    pred_class_value = pred_classes.item()
    # 返回检测结果给前端
    socketio.emit('detection_result', {'confidence': value, 'classification': pred_class_value,'url':img_url})
    return 'picture ok'

# 图片预测age
@app.route('/detect_picture_age', methods=['POST'])
def detect_picture_age():
    # 获取上传的图片文件
    file = request.files['image']
    # 边距
    padding = 20
    # 读取图片数据
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        #np.argmax()是用于取得数组中每一行或者每一列的的最大值。常用于机器学习中获取分类结果、计算精确度等。
        #np.max:返回数组中的最大值；  np.argmax: 返回数组中最大值坐标
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
  
        # 去除括号
        age = age.replace("(", "").replace(")", "")
        label = "{} years old".format(age)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
     # 这里假设直接返回原图的 base64 编码
    _, img_encoded = cv2.imencode('.jpg', frameFace)
    img_base64 = base64.b64encode(img_encoded)
    img_url = 'data:image/jpeg;base64,' + img_base64.decode('utf-8')
    print('ok,picture')
    # 返回检测结果给前端
    socketio.emit('detection_result', {'confidence': '222', 'classification':' pred_class_value','url':img_url})
    return 'picture ok'

# 图片预测gender
@app.route('/detect_picture_gender', methods=['POST'])
def detect_picture_gender():
    # 获取上传的图片文件
    file = request.files['image']
    # 边距
    padding = 20
    # 读取图片数据
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        #np.argmax()是用于取得数组中每一行或者每一列的的最大值。常用于机器学习中获取分类结果、计算精确度等。
        #np.max:返回数组中的最大值；  np.argmax: 返回数组中最大值坐标
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
  
        # 去除括号
        age = age.replace("(", "").replace(")", "")
        label = "{}".format(gender)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
     # 这里假设直接返回原图的 base64 编码
    _, img_encoded = cv2.imencode('.jpg', frameFace)
    img_base64 = base64.b64encode(img_encoded)
    img_url = 'data:image/jpeg;base64,' + img_base64.decode('utf-8')
    print('ok,picture')
    # 返回检测结果给前端
    socketio.emit('detection_result', {'confidence': '222', 'classification':' pred_class_value','url':img_url})
    return 'picture ok'
              
    
# 上传视频检测人脸反欺诈
@app.route('/upload_video',methods=['GET','POST'])
def upload_video():
    if 'video' not in request.files:
        print('NO video')
        return jsonify({'error': 'No video provided'}), 400
    global cap
    video_file = request.files['video']
    video_file_path = 'uploaded_video.mp4'  # 保存上传的视频文件
    video_file.save(video_file_path)
    
    cap = cv2.VideoCapture(video_file_path)
   
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Cannot open video file")
    while cap.isOpened():
        start_time = time.time()  # 记录开始时间
        frame_count = 0  # 重置帧数计数器
        ret, frame = cap.read()
        
        # 如果成功读取到帧
        if ret:
            # 获取视频帧的高度
            height = frame.shape[0]
            image_bbox = detector.get_bbox(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(image)
            # [N, C, H, W]
            image = data_transform(img_pil)
            # expand batch dimension
            image = torch.unsqueeze(image, dim=0)
            
            with torch.no_grad():
                # predict class
                output= model(image.to(device))
                pred_classes = torch.max(output, dim=1)[1]

                aaa = torch.max(output, dim = 1)[0].item() - torch.min(output, dim = 1)[0].item()
                value = 1 - np.exp(-aaa)
                if pred_classes == 1:
                    result_text = "RealFace Score: {:.3f}".format(value)
                    color = (255, 0, 0)
                else:
                    result_text = "FakeFace Score: {:.3f}".format(value)
                    color = (0, 0, 255)
                cv2.rectangle(
                    frame,
                    (image_bbox[0], image_bbox[1]),
                    (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                    color, 2)
                cv2.putText(
                    frame,
                    result_text,
                    (image_bbox[0], image_bbox[1] - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5*frame.shape[0]/1024, color)
                    
                # 计算帧率
            elapsed_time = time.time() - start_time  # 计算时间间隔
            frame_count += 1  # 帧数加1
            fps = frame_count / elapsed_time  # 计算帧率
            cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode()  # 将字节对象转换为字符串
            img_url = 'data:image/jpeg;base64,' + img_base64

            # 提取pred_classes中的值并转换为Python标量
            
            pred_class_value = pred_classes.item()
            # 使用 socket 返回结果给前端
            socketio.emit('detection_result', {'confidence': value, 'classification': pred_class_value, 'url': img_url})
            # 返回图像URL给前端
            # return jsonify({'image_url': img_url})
        else:
        # 如果已经读取到视频的最后一帧，则退出循环
            break    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'message': 'Video uploaded successfully'})

# 上传视频检测年龄
@app.route('/upload_video_age',methods=['GET','POST'])
def upload_video_age():
    if 'video' not in request.files:
        print('NO video')
        return jsonify({'error': 'No video provided'}), 400
    global cap
    video_file = request.files['video']
    video_file_path = 'uploaded_video.mp4'  # 保存上传的视频文件
    video_file.save(video_file_path)
    
    cap = cv2.VideoCapture(video_file_path)
    padding = 20
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Cannot open video file")
    while cap.isOpened():
        start_time = time.time()  # 记录开始时间
        frame_count = 0  # 重置帧数计数器
        ret, frame = cap.read()
        
        # 如果成功读取到帧
        if ret:
            frameFace, bboxes = getFaceBox(faceNet, frame)
            if not bboxes:
                print("No face Detected, Checking next frame")
                continue
            for bbox in bboxes:
                # print(bbox)
                face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                #np.argmax()是用于取得数组中每一行或者每一列的的最大值。常用于机器学习中获取分类结果、计算精确度等。
                #np.max:返回数组中的最大值；  np.argmax: 返回数组中最大值坐标
                gender = genderList[genderPreds[0].argmax()]

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                
                # 去除括号
                age = age.replace("(", "").replace(")", "")
                label = "{} years old".format(age)
                # 设置字体大小
                font_scale = 1.5

                # 在图像上绘制文字
                cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2, cv.LINE_AA)

            
            elapsed_time = time.time() - start_time  # 计算时间间隔
            frame_count += 1  # 帧数加1
            fps = frame_count / elapsed_time  # 计算帧率
            cv2.putText(frameFace, "FPS: {:.2f}".format(fps), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frameFace)
            img_base64 = base64.b64encode(buffer).decode()  # 将字节对象转换为字符串
            img_url = 'data:image/jpeg;base64,' + img_base64

            # 使用 socket 返回结果给前端
            socketio.emit('detection_result', {'confidence': "value", 'classification': "pred_class_value", 'url': img_url})
            # 返回图像URL给前端  
        else:
        # 如果已经读取到视频的最后一帧，则退出循环
            break         
      # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'message': 'Video uploaded successfully'})
        
# 上传视频检测性别
@app.route('/upload_video_gender',methods=['GET','POST'])
def upload_video_gender():
    if 'video' not in request.files:
        print('NO video')
        return jsonify({'error': 'No video provided'}), 400
    global cap
    video_file = request.files['video']
    video_file_path = 'uploaded_video.mp4'  # 保存上传的视频文件
    video_file.save(video_file_path)
    
    cap = cv2.VideoCapture(video_file_path)
    padding = 20
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Cannot open video file")
    while cap.isOpened():
        start_time = time.time()  # 记录开始时间
        frame_count = 0  # 重置帧数计数器
        ret, frame = cap.read()
        
        # 如果成功读取到帧
        if ret:
            frameFace, bboxes = getFaceBox(faceNet, frame)
            if not bboxes:
                print("No face Detected, Checking next frame")
                continue
            for bbox in bboxes:
                # print(bbox)
                face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                #np.argmax()是用于取得数组中每一行或者每一列的的最大值。常用于机器学习中获取分类结果、计算精确度等。
                #np.max:返回数组中的最大值；  np.argmax: 返回数组中最大值坐标
                gender = genderList[genderPreds[0].argmax()]

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                
                label = "{} ".format(gender)
                # 设置字体大小
                font_scale = 1.5

                # 在图像上绘制文字
                cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2, cv.LINE_AA)

            
            elapsed_time = time.time() - start_time  # 计算时间间隔
            frame_count += 1  # 帧数加1
            fps = frame_count / elapsed_time  # 计算帧率
            cv2.putText(frameFace, "FPS: {:.2f}".format(fps), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frameFace)
            img_base64 = base64.b64encode(buffer).decode()  # 将字节对象转换为字符串
            img_url = 'data:image/jpeg;base64,' + img_base64

            # 使用 socket 返回结果给前端
            socketio.emit('detection_result', {'confidence': "value", 'classification': "pred_class_value", 'url': img_url})
            # 返回图像URL给前端  
        else:
        # 如果已经读取到视频的最后一帧，则退出循环
            break         
      # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'message': 'Video uploaded successfully'})       
        
        
        
        
        

        
if __name__ == '__main__':
    # app.run(debug=True,host='0.0.0.0',port="8080")
    socketio.run(app)
