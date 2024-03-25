from flask import Flask, request, Response, render_template, send_from_directory,url_for,redirect,jsonify
import cv2
import os
import sys
import base64
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import argparse
import warnings
import time
import random
from main.anti_spoof_predict import AntiSpoofPredict
from main.generate_patches import CropImage
from main.utility import parse_model_name
from flask_socketio import SocketIO
from io import BytesIO
from PIL import Image
from engineio.payload import Payload
warnings.filterwarnings('ignore')

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

# @socketio.on('video_frame')
# def handle_video_frame(blob):
#     # 在这里处理接收到的视频帧数据，您可以将其保存到文件、进行处理等操作
#     print('reveiced video')
#     pass


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
    # 在这里执行图片检测操作
    model_test = AntiSpoofPredict(0)
    video_cropper = CropImage()
    image_bbox = model_test.get_bbox(img_array)
    prediction = np.zeros((1, 3))
    model_dir="./resources/anti_spoof_models"
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": img_array,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img_array = video_cropper.crop(**param)
        prediction += model_test.predict(
            img_array, os.path.join(model_dir, model_name))
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        
        label1 = 'true'
    else:
        
        label1 = 'false'
    
    print('ok')
    socketio.emit('detection_result_withoutpicture', {'confidence': value, 'classification': label1})



    
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

# # 使用摄像头
# @app.route('/video_capture',methods=['GET','POST'])
# def video_capture():
    
#     # 接收图像数据
#     image_file = request.files['image']
#     image_blob = image_file.read()
#     # 将图像数据转换为图像对象
#     image = Image.open(io.BytesIO(image_blob))
#     capture_by_frames(image_data=image)
#     return 'nihao,摄像头'

# # 摄像头检测
# def capture_by_frames(image_data,model_dir="./resources/anti_spoof_models"):
#     model_test = AntiSpoofPredict(0)
#     video_cropper = CropImage()
#     # 将接收到的图像数据解码为图像数组
#     nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     frame_count = 0  # 重置帧数计数器
#     frame_bbox = model_test.get_bbox(frame)
#     prediction = np.zeros((1, 3))
#     for model_name in os.listdir(model_dir):
#         h_input, w_input, model_type, scale = parse_model_name(model_name)
#         param = {
#             "org_img": frame,
#             "bbox": frame_bbox,
#             "scale": scale,
#             "out_w": w_input,
#             "out_h": h_input,
#             "crop": True,
#         }
#         if scale is None:
#             param["crop"] = False
#         img = video_cropper.crop(**param)
#         prediction += model_test.predict(
#             img, os.path.join(model_dir, model_name))
#     label = np.argmax(prediction)
#     value = prediction[0][label] / 2
#     if label == 1:
#         result_text = "RealFace Score: {:.2f}".format(value)
#         color = (255, 0, 0)
#         label1 = 'ture face'
#     else:
#         result_text = "FakeFace Score: {:.2f}".format(value)
#         color = (0, 0, 255)
#         label1 = 'false face'
#     cv2.rectangle(
#         frame, (frame_bbox[0], frame_bbox[1]),
#         (frame_bbox[0] + frame_bbox[2], frame_bbox[1] + frame_bbox[3]),
#         color, 2)
#     cv2.putText(frame, result_text, (frame_bbox[0], frame_bbox[1] - 5),
#                 cv2.FONT_HERSHEY_COMPLEX, 0.5 * h_input / 1024, color)

#     # 计算帧率
#     cv2.putText(frame, "FPS: N/A", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     _, buffer = cv2.imencode('.jpg', frame)
#     img_base64 = base64.b64encode(buffer).decode()  # 将字节对象转换为字符串
#     img_url = 'data:image/jpeg;base64,' + img_base64
#     # 使用 socket 返回结果给前端
#     socketio.emit('detection_result', {'confidence': value, 'classification': label1, 'url': img_url})


    
    
    
    
    
    
    
    
    
    
    
    
    
    

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
    
# 上传视频检测
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
    model_test = AntiSpoofPredict(0)
    video_cropper = CropImage()
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Cannot open video file")
    model_dir="./resources/anti_spoof_models"
    while cap.isOpened():
        start_time = time.time()  # 记录开始时间
        frame_count = 0  # 重置帧数计数器
        ret, frame = cap.read()
        
        # 如果成功读取到帧
        if ret:
            # 获取视频帧的高度
            height = frame.shape[0]
            frame_bbox = model_test.get_bbox(frame)
            prediction = np.zeros((1, 3))
            for model_name in os.listdir(model_dir):
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": frame,
                    "bbox": frame_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                img = video_cropper.crop(**param)
                prediction += model_test.predict(
                    img, os.path.join(model_dir, model_name))
            label = np.argmax(prediction)
            value = prediction[0][label] / 2
            if label == 1:
                result_text = "RealFace Score: {:.2f}".format(value)
                color = (255, 0, 0)
                label1 = 'true'
            else:
                result_text = "FakeFace Score: {:.2f}".format(value)
                color = (0, 0, 255)
                label1 = 'flase'
            cv2.rectangle(
                frame, (frame_bbox[0], frame_bbox[1]),
                (frame_bbox[0] + frame_bbox[2], frame_bbox[1] + frame_bbox[3]),
                color, 2)
            cv2.putText(frame, result_text, (frame_bbox[0], frame_bbox[1] - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5 * height / 1024, color)
            

            # 计算帧率
            elapsed_time = time.time() - start_time  # 计算时间间隔
            frame_count += 1  # 帧数加1
            fps = frame_count / elapsed_time  # 计算帧率
            cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer)
            img_url = 'data:image/jpeg;base64,' + img_base64.decode('utf-8')
            socketio.emit('detection_result',{'confidence':value,'classification':label1,'url':img_url})
            i+=1
            print(i)
            # 返回图像URL给前端
            # return jsonify({'image_url': img_url})
        else:
        # 如果已经读取到视频的最后一帧，则退出循环
            break
        
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'message': 'Video uploaded successfully'})

# 停止视频检测
@app.route('/stop_video',methods=['POST','GET'])
def stop_video():
    print('stop')
    if cap is not None:
        cap.release()  # 释放视频资源
        
        return 'video released successfully'
    else:
        return 'video is not in use'
    
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
    
    # 获取图像的宽度和高度
    height, width, channels = img.shape
    
    # 在这里执行图片检测操作
    model_test = AntiSpoofPredict(0)
    video_cropper = CropImage()
    image_bbox = model_test.get_bbox(img)
    prediction = np.zeros((1, 3))
    model_dir="./resources/anti_spoof_models"
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": img,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = video_cropper.crop(**param)
        prediction += model_test.predict(
            img, os.path.join(model_dir, model_name))
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
        label1 = 'true'
    else:
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
        label1 = 'false'
    cv2.rectangle(
        img_copy, (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    font_scale = 0.5  # 修改字体大小的比例因子
    cv2.putText(img_copy, result_text, (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, font_scale * height / 1024, color)

    # 这里假设直接返回原图的 base64 编码
    _, img_encoded = cv2.imencode('.jpg', img_copy)
    img_base64 = base64.b64encode(img_encoded)
    img_url = 'data:image/jpeg;base64,' + img_base64.decode('utf-8')
    print('ok')
    # 返回检测结果给前端
    socketio.emit('detection_result_picture', {'confidence': value, 'classification': label1,'url':img_url})
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
