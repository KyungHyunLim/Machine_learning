import os
import cv2
import numpy as np

def classes(str):
    if str == 'dog':
        return 0
    elif str == 'elephant':
        return 1
    elif str == 'giraffe':
        return 2
    elif str == 'guitar':
        return 3
    elif str == 'horse':
        return 4
    elif str == 'house':
        return 5
    else:
        return 6

def data_load(path, num_of_class):
    X = []
    Y = []
    for root, dirs, files in os.walk(path):
        label = [0 for i in range(num_of_class)]
        classes_idx = classes(root.split('\\')[-1])
        label[classes_idx] = 1

        for fname in files:
            full_path = os.path.join(root, fname)
            img = cv2.imread(full_path)
            
            X.append(img/256)
            Y.append(label)
        
    X = np.array(X)
    Y = np.array(Y)
    
    np.save('./img_data_X.npy', X)
    np.save('./img_data_Y.npy', Y)
    print(Y)
    
    
if __name__ == '__main__':
    data_load('./train', 7)