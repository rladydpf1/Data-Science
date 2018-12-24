import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

X = np.zeros((160, 2000), dtype=float)  # 학습 데이터
Xt = np.zeros((240, 2000), dtype=float) # 테스트 데이터
Y = np.zeros((160,1), dtype=int)    # 학습 클래스 레이블
Yt = np.zeros((240,1), dtype=int)   # 테스트 클래스 레이블

# load train image into a numpy matrix
for i, img_file in enumerate(glob.glob('./data/ORL/train/*.png')):
    img = mpimg.imread(img_file)
    X[i,:] = np.reshape(img, (1,2000))
# display last image
plt.imshow(img, cmap='gray')
plt.show()

# load test image into a numpy matrix
for i, img_file in enumerate(glob.glob('./data/ORL/test/*.png')):
    img = mpimg.imread(img_file)
    Xt[i,:] = np.reshape(img, (1,2000))

# set class label
for i in range(40):
    Y[4*i:4*(i+1),0] = i+1
    Yt[6*i:6*(i+1),0] = i+1