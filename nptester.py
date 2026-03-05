
import numpy as np



def test():
    y1 = np.zeros(5)
    y2 = np.ones(5)
    y = np.concatenate((y1, y2), axis=0).reshape(-1, 1)
    print(y)

def test_split():
    arr = [1,2,3,4,5,6,7,8,9,10]
    X = []
    y = []
    step = 6
    for i in range(len(arr) - step):
        X.append([e for e in arr[i : i + step]])
        y.append(arr[i + step])
    print(X, y)

if __name__ == "__main__":
    test_split()
