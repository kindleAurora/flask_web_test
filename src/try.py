import os,random
# 定义照片文件夹路径
photo_folder = '9'
photos = [os.path.join(photo_folder,file)for file in os.listdir(photo_folder)if file.endswith(('.jpg','.png','.jpeg'))]
random_photo = random.choice(photos)
print(random_photo)