import os
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

dir_data = './datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames

## 30frame 중 24개는 train
#프레임의 개수는 어찌 확인??
nframe_train = 24
nframe_val = 3
nframe_test = 3
#Q 이거 스키트런으로 나눌 수 있단거지?
#홀드아웃으로 나누는 방식들 조사
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## 프레임에 대하여 랜덤 엑세스를 실행
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

##
offset_nframe = 0 #what is the offset?

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%3d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%3d.npy' % i), input_)

##
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()

