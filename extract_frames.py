#from imageio.core.functions import RETURN_BYTES
#!/usr/bin/env python
# coding=utf-8

import os
import cv2

# 视频路径和图片保存路径
videos_path = r"./video"
images_path = r"./data"
if not os.path.exists(images_path):
    os.makedirs(images_path)

# 遍历读取视频文件---支持多级目录下的视频文件遍历
i = 0
file_count = 0
for root, dirs, files in os.walk(videos_path):
    for file_name in files:
        file_count += 1
        i += 1
        # os.mkdir(images_path + '/' + '0324_%d' % i)
        # img_full_path = os.path.join(images_path, '0324_%d' % i) + '/'
        os.mkdir(images_path + '/' + file_name.split('.')[0])
        #os.mkdir(images_path + '/' + str(i))
        img_full_path = os.path.join(images_path, file_name.split('.')[0]) + '/'
        #img_full_path = os.path.join(images_path, str(i)) + '/'
        videos_full_path = os.path.join(root, file_name)
        cap = cv2.VideoCapture(videos_full_path)
        print('\n开始处理第 ', str(i), ' 个视频：'+file_name)
        if cap.isOpened():
          #current_frame = 0
          frame_count = 0
          ret = True
          while ret:
            ret, frame = cap.read()
            if ret:
              name = img_full_path + "%d.jpg" % (frame_count)
              #name = img_full_path + "frame%03d.jpg" % (frame_count+1)
              print(f"Creating file... {name}")
              cv2.imwrite(name, frame)
              #frame.append(name)
            frame_count += 1
          #cap.release()
        #cv2.destroyAllWindows()
      
print('\n一共 ', str(file_count), ' 个视频,', '已全部处理完毕！')

