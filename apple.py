# 通过样本图片建立模型, 对其他图片进行判断
import os
from collections import Counter

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def perpare_data():
    path = "data/imgs/apple/original"
    dst_path = "data/imgs/apple/gen"
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.02,
        horizontal_flip=True,
        vertical_flip=True,
    )
    gen = datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=2,
        save_to_dir=dst_path,
        save_prefix="gen",
        save_format="jpg",
    )
    for _ in range(100):
        gen.__next__()


def load_imgs(folder="data/imgs/apple/training"):
    imgs = []
    for filename in os.listdir(folder):
        if os.path.splitext(filename)[1] == ".jpg":
            img_path = os.path.join(folder, filename)
            imgs.append(img_path)
    return imgs


def img_features(img_path, model):
    img = img_to_array(load_img(img_path, target_size=(224, 224)))
    x = model.predict(preprocess_input(np.expand_dims(img, axis=0)))
    return x.reshape(1, 7 * 7 * 512)


def load_img_features(imgs, model):
    features = np.zeros([len(imgs), 7 * 7 * 512])
    for i in range(len(imgs)):
        features[i] = img_features(imgs[i], model)
    print("load img features:", features.shape, features)
    return features


def show_result(imgs, y_test_pred, normal_val=0):
    plt.figure(figsize=(8, 8))
    for i in range(4):
        for j in range(3):
            img = load_img(imgs[i * 3 + j])
            plt.subplot(4, 3, i * 3 + j + 1)
            plt.imshow(img)
            plt.title("apple" if y_test_pred[i * 3 + j] == normal_val else "other")
            plt.axis("off")
    plt.show()


def knn_imgs(X, X_test):
    # 基于vgg提取的特征,进行kmeans聚类
    cnn_kmeans = KMeans(n_clusters=2, max_iter=1000, random_state=99)
    cnn_kmeans.fit(X)
    y_pred = cnn_kmeans.predict(X)
    # 实际值应该是210一类 200一类, 但knn识别出来1xx一类 1xx一类
    print("cnn_pred:", y_pred, Counter(y_pred))

    test_pred = cnn_kmeans.predict(X_test)
    print("knn_test_pred:", test_pred, Counter(test_pred))
    return test_pred


def mean_shift_imgs(X, X_test):
    # 基于vgg提取的特征,进行mean shift聚类
    ms = MeanShift(bandwidth=estimate_bandwidth(X, n_samples=100))
    ms.fit(X)
    y_pred = ms.predict(X)
    print("ms_pred:", y_pred, Counter(y_pred))
    test_pred = ms.predict(X_test)
    print("ms_test_pred:", test_pred, Counter(test_pred))
    return test_pred


def pca(X, X_test):
    """
    对数据进行PCA降维,降维后给k_means和mean shift聚类
    """
    # 数据标准化
    std = StandardScaler()
    X_norm = std.fit_transform(X)
    X_test_norm = std.transform(X_test)
    # 进行PCA降维,期望从224*224*3 降到 200维, 即希望保留200个主成分
    pca = PCA(n_components=200)
    X_reduced = pca.fit_transform(X_norm)
    # 查看降维后数据的方差占比
    var_ratio = pca.explained_variance_ratio_
    print("X_train variance ratio: ", sum(var_ratio))
    X_test_reduced = pca.transform(X_test_norm)
    var_ratio_test = pca.explained_variance_ratio_
    print("X_test variance ratio: ", sum(var_ratio_test))   
    print("X_reduced:", X_reduced.shape)
    print("X_test_reduced:", X_test_reduced.shape)
    return X_reduced, X_test_reduced


if __name__ == "__main__":
    vgg = VGG16(weights="imagenet", include_top=False)
    X = load_img_features(load_imgs(), vgg)
    X_test = load_img_features(load_imgs("data/imgs/apple/testing"), vgg)
    # y_test_pred = knn_imgs(X, X_test)
    # y_test_pred = mean_shift_imgs(X, X_test)
    X_reduced, X_test_reduced = pca(X, X_test)
    y_test_pred = mean_shift_imgs(X_reduced, X_test_reduced)
    show_result(load_imgs("data/imgs/apple/testing"), y_test_pred)
