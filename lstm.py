# 字符级LSTM文本生成


def formate_doc():
    newtxt = []
    for line in open("data/daodejing.txt", "r").readlines():
        line = line.replace("\r", "").replace("\n", "").lstrip("0123456789. ")
        if not line:
            continue
        newtxt.append(line)
    open("data/daodejing1.txt", "w").writelines("\n".join(newtxt))


data = open("data/daodejing.txt", "r").read().replace("\n", "").replace("\r", "") * 2
# 字符去重处理
letters = list(set(data))
num_letters = len(letters)

print(">>>", data[:100], f"...[{len(data)}]", "letters:", num_letters)

# 建立字典
int_to_char = {a: b for a, b in enumerate(letters)}
char_to_int = {b: a for a, b in enumerate(letters)}

# 20个字符预测第21个
time_step = 30

import numpy as np
from keras.utils import to_categorical


def extract_data(data, slide):
    x = []
    y = []
    for i in range(len(data) - slide):
        x.append([a for a in data[i : i + slide]])
        y.append(data[i + slide])
    return x, y


def char_to_int_data(x, y, char_to_int):
    x_to_int = []
    y_to_int = []
    for i in range(len(x)):
        x_to_int.append([char_to_int[ch] for ch in x[i]])
        y_to_int.append([char_to_int[ch] for ch in y[i]])
    return x_to_int, y_to_int


def data_processing(data, slide, num_letters, char_to_int):
    char_data = extract_data(data, slide)
    int_data = char_to_int_data(char_data[0], char_data[1], char_to_int)

    input = int_data[0]
    output = list(np.array(int_data[1]).flatten())
    input_reshaped = np.array(input).reshape(len(input), slide)
    new = np.random.randint(
        0, 10, size=[input_reshaped.shape[0], input_reshaped.shape[1], num_letters]
    )
    for i in range(input_reshaped.shape[0]):
        for j in range(input_reshaped.shape[1]):
            new[i, j, :] = to_categorical(input_reshaped[i, j], num_classes=num_letters)
    return new, output


X, y = data_processing(data, time_step, num_letters, char_to_int)
print(f"total data shape: X={X.shape}, y={len(y)}")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=6)
y_train_category = to_categorical(y_train, num_letters)


from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam


model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=time_step, activation="relu"))
model.add(Dense(units=num_letters, activation="softmax"))
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

model.fit(X_train, y_train_category, epochs=29, batch_size=37)

y_train_pred = model.predict(X_train).argmax(axis=-1)
y_test_pred = model.predict(X_test).argmax(axis=-1)
from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"training accuracy: {train_accuracy:.4f}", f"test accuracy: {test_accuracy:.4f}")

new_letters = "小国寡民。使有什伯之器而不用。使民重死而不远徙。虽有舟舆无所乘之。虽有甲兵无所陈之。使民复结绳而用之。甘其食、美其服、安其居、乐其俗。邻国相望，鸡犬之声相闻。民至老死不相往来。"
X_new, y_new = data_processing(new_letters, time_step, num_letters, char_to_int)
y_new_pred = model.predict(X_new).argmax(axis=-1)
print(">>>:", new_letters)
print("<<<:", "。" * time_step, "".join([int_to_char[i] for i in y_new_pred]))
