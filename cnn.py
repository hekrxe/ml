from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    training_set = datagen.flow_from_directory(
        "data/imgs/training_set",
        target_size=(50, 50),
        batch_size=128,  # 增大batch_size加速训练
        class_mode="binary",
    )
    print("training_set:", training_set)

    model = Sequential()
    model.add(Input(shape=(50, 50, 3)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # 减少一个卷积层，简化模型
    model.add(Flatten())
    # 减少全连接层神经元数量，从128降到64
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.001),  # 降低学习率，更稳定
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 添加早停回调，避免过训练
    early_stop = EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True  # 3个epoch没有改善就停止
    )

    model.fit(
        training_set,
        epochs=30,  # 减少epochs，配合早停
        callbacks=[early_stop], 
    )

    loss, accuracy = model.evaluate(training_set)
    print("training_set. loss:", loss, "accuracy:", accuracy)

    test_set = datagen.flow_from_directory(
        "data/imgs/test_set",
        target_size=(50, 50),
        batch_size=32,
        class_mode="binary",
    )
    loss, accuracy = model.evaluate(test_set)
    print("test_set. loss:", loss, "accuracy:", accuracy)

    testing_data = datagen.flow_from_directory(
        "data/imgs/testing_data",
        target_size=(50, 50),
        batch_size=32,
        class_mode="binary",
    )
    loss, accuracy = model.evaluate(testing_data)
    print("testing_data. loss:", loss, "accuracy:", accuracy)

    from keras.preprocessing.image import load_img, img_to_array

    cat = (img_to_array(load_img("cat.jpg", target_size=(50, 50))) / 255.0).reshape(
        1, 50, 50, 3
    )
    prediction = model.predict(cat)
    print(
        "cat. prediction: ",
        prediction,
        ("cat" if prediction[0][0] > 0.5 else "None"),
    )

    dog = (img_to_array(load_img("dog.jpg", target_size=(50, 50))) / 255.0).reshape(
        1, 50, 50, 3
    )
    prediction = model.predict(dog)
    print(
        "dog. prediction: ",
        prediction,
        ("dog" if prediction[0][0] < 0.5 else "None"),
    )
