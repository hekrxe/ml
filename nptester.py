
import numpy as np



def test():
    y1 = np.zeros(5)
    y2 = np.ones(5)
    y = np.concatenate((y1, y2), axis=0).reshape(-1, 1)
    print(y)



if __name__ == "__main__":
    test()
