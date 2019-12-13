import os
import numpy
import matplotlib.pyplot as plt
import pickle

Path = "D:\DataSet\Anime_face_dataset"

image_file = os.listdir(Path)
label_0_path = []
label_1_path = []
label_2_path = []

i = 0
for f in image_file:
    if image_file[i][-5] == '0':
        label_0_path.append(f)
    elif image_file[i][-5] == '1':
        label_1_path.append(f)
    elif image_file[i][-5] == '2':
        label_2_path.append(f)
    i += 1

print(label_0_path[0])
print(label_1_path[0])
print(label_2_path[0])

image_data_0 = []
image_data_1 = []
image_data_2 = []

for path in label_0_path:
    image_data_0.append(plt.imread(Path + "\\" + path))

for path in label_1_path:
    image_data_1.append(plt.imread(Path + "\\" + path))

for path in label_2_path:
    image_data_2.append(plt.imread(Path + "\\" + path))

plt.imshow(image_data_0[0])
plt.imshow(image_data_1[0])
plt.imshow(image_data_2[0])

fw_0 = open("D:\DataSet\\test_data\image_0.txt","wb")
pickle.dump(image_data_0, fw_0)
fw_0.close()

fw_1 = open("D:\DataSet\\test_data\image_1.txt","wb")
pickle.dump(image_data_1, fw_1)
fw_1.close()

fw_2 = open("D:\DataSet\\test_data\image_2.txt","wb")
pickle.dump(image_data_2, fw_2)
fw_2.close()
