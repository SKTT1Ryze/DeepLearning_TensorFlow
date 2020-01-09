import os
import numpy
import tensorflow as tf
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
Path = "D:\DataSet\Anime_face_dataset"

image_file = os.listdir(Path)

image_data = []

for f in image_file:
    x = tf.io.read_file(Path + "\\" + f)
    x = tf.image.decode_jpeg(x, channels=3)
    image_data.append(x)

fw = open("D:\DataSet\\test_data\Anime_Image_tf.txt", "wb")
pickle.dump(image_data, fw)
fw.close()
