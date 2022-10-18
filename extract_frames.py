#from imageio.core.functions import RETURN_BYTES
#!/usr/bin/env python
# coding=utf-8

import os
import cv2

# video path and output frames path
videos_path = r"./video"
images_path = r"./data"
if not os.path.exists(images_path):
    os.makedirs(images_path)

# get the frames of each video
i = 0
file_count = 0
for root, dirs, files in os.walk(videos_path):
    for file_name in files:
        file_count += 1
        i += 1
        os.mkdir(images_path + '/' + file_name.split('.')[0])
        img_full_path = os.path.join(images_path, file_name.split('.')[0]) + '/'
        videos_full_path = os.path.join(root, file_name)
        cap = cv2.VideoCapture(videos_full_path)
        print('\nbegin to deal with the ', str(i), '-th video ：'+file_name)
        if cap.isOpened():
          frame_count = 0
          ret = True
          while ret:
            ret, frame = cap.read()
            if ret:
              name = img_full_path + "%d.jpg" % (frame_count)
              print(f"Creating file... {name}")
              cv2.imwrite(name, frame)
            frame_count += 1
         
print('\nThere are ', str(file_count), ' videos,', 'and they are all sampled into frames.')

