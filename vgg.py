# vgg 猫狗识别

import os

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Input


def model_process(img_path, model):
    img = img_to_array(load_img(img_path, target_size=(224, 224)))
    features = model.predict(preprocess_input(np.expand_dims(img, axis=0)))
    return features.reshape(1, 7 * 7 * 512)


def get_imgs(folder):
    imgs = []
    for filename in os.listdir(folder):
        if os.path.splitext(filename)[1] == ".jpg":
            imgs.append(os.path.join(folder, filename))
    print(len(imgs), imgs[0])
    return imgs


def get_features(imgs, model):
    features = np.zeros([len(imgs), 7 * 7 * 512])
    for i in range(len(imgs)):
        features[i] = model_process(imgs[i], model)
    return features


def load_Xy(model):
    cats = get_imgs("data/imgs/training_data/cats")
    dogs = get_imgs("data/imgs/training_data/dogs")
    cat_features = get_features(cats, model)
    dog_features = get_features(dogs, model)

    print("cat features:", cat_features.shape, " dog features:", dog_features.shape)

    # 创建标签
    y1 = np.zeros(len(cats))
    y2 = np.ones(len(dogs))

    X = np.concatenate((cat_features, dog_features), axis=0)
    y = np.concatenate((y1, y2), axis=0).reshape(-1, 1)
    print("X:", X.shape, " y:", y.shape)
    return X, y


if __name__ == "__main__":
    # 使用VGG16模型提取图像特征,再根据特征建立mlp模型,实现猫狗图片识别
    vgg = VGG16(weights="imagenet", include_top=False)
    X, y = load_Xy(vgg)
    # split train test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=99
    )
    print("X_train:", X_train.shape, " y_train:", y_train.shape)
    print("X_test:", X_test.shape, " y_test:", y_test.shape)

    mlp = Sequential()
    mlp.add(Input(shape=(7 * 7 * 512,)))
    mlp.add(Dense(units=10, activation="relu"))
    mlp.add(Dense(units=1, activation="sigmoid"))
    mlp.summary()

    mlp.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    mlp.fit(X_train, y_train, epochs=50, batch_size=32)

    y_train_pred = (mlp.predict(X_train) > 0.5).astype("int32")
    print("train acc: ", accuracy_score(y_train, y_train_pred))

    y_test_pred = (mlp.predict(X_test) > 0.5).astype("int32")
    print("test acc: ", accuracy_score(y_test, y_test_pred))

    loss, acc = mlp.evaluate(X_test, y_test)
    print("test data. loss:", loss, " acc:", acc)

    # 0 猫, 1 狗
    for i in ["cat.jpg", "dog.jpg"]:
        img = img_to_array(load_img(i, target_size=(224, 224)))
        x = preprocess_input(np.expand_dims(img, axis=0))
        features = vgg.predict(x).reshape(1, 7 * 7 * 512)
        y_pred = (mlp.predict(features) > 0.5).astype("int32")
        print("testing: ", i, features.shape, "pred: ", y_pred)
