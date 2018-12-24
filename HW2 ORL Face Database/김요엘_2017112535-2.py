import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cluster import KMeans

X = np.zeros((160, 2000), dtype=float)  # 학습 데이터
Xt = np.zeros((240, 2000), dtype=float)  # 테스트 데이터
 # 학습 클래스 레이블
Yt = np.zeros((240, 1), dtype=int)  # 테스트 클래스 레이블

# load train image into a numpy matrix
for i, img_file in enumerate(glob.glob('data/train/*.png')):
    img = mpimg.imread(img_file)
    X[i, :] = np.reshape(img, (1, 2000))

# load test image into a numpy matrix
for i, img_file in enumerate(glob.glob('data/test/*.png')):
    img = mpimg.imread(img_file)
    Xt[i, :] = np.reshape(img, (1, 2000))

listY = []
# set class label
for i in range(40):
    listY.append(i + 1)
    listY.append(i + 1)
    listY.append(i + 1)
    listY.append(i + 1)
    Yt[6 * i:6 * (i + 1), 0] = i + 1
Y = np.array(listY, ndmin = 1)

K = 1
accu_pca = []
accu_lda = []

# PCA, 1차원 ~ 39차원
for i in range(39):
    pcai = PCA(n_components=(i + 1))
    neigh = KNN(n_neighbors=K)
    neigh.fit(pcai.fit_transform(X), Y)
    Xt_new = pcai.transform(Xt)
    accu_pca.append(neigh.score(Xt_new, np.ravel(Yt)))

# LDA, 1차원 ~ 39차원
for i in range(39):
    ldai = LDA(n_components=(i + 1))
    ldai.fit(X, Y)
    neigh = KNN(n_neighbors=K)
    neigh.fit(ldai.transform(X), Y)
    Xt_new = ldai.transform(Xt)
    accu_lda.append(neigh.score(Xt_new, np.ravel(Yt)))

# KNN 한 것을 시각화
df = pd.DataFrame({'x': range(1, 40), 'PCA': accu_pca, 'LDA': accu_lda})
plt.plot('x', 'PCA', data = df, color='blue', label ='PCA')
plt.plot('x', 'LDA', data = df, color='red', label = 'LDA')
plt.title("PCA vs. LDA KNN ACCURACY")
plt.xlabel("Number of Features")
plt.ylabel("Accuracy(%)")
plt.legend()
plt.show()

# K-means 군집화 수행
K = 40
print("군집화 결과 - RAW 데이터")
clustering = KMeans(n_clusters=K)
clustering.fit(X)
print(clustering.predict(X))

print("군집화 결과 - PCA")
pcai = PCA(n_components=(39))
X_pca = pcai.fit_transform(X)
clustering = KMeans(n_clusters=K)
clustering.fit(X_pca)
print(clustering.predict(X_pca))

print("군집화 결과 - LDA")
ldai = LDA(n_components=39)
ldai.fit(X, Y)
X_lda = ldai.transform(X)
clustering = KMeans(n_clusters=K)
clustering.fit(X_lda)
print(clustering.predict(X_lda))