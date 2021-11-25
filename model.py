import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
import imageio
import cv2 
import os
from PIL import Image
from scipy import ndimage
from scipy import misc
import pickle

train_messy = "./input/images/train/messy/"
train_clean= "./input/images/train/clean"
test_messy= "./input/images/val/messy"
test_clean= "./input/images/val/clean"

def load_dataset():
    
    train_set_x_orig_list,train_set_y_list, test_set_x_orig_list, test_set_y_list=[],[],[],[]

    for image in (os.listdir(train_messy)): 
        path = os.path.join(train_messy, image)
        img = cv2.imread(path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_set_x_orig_list.append(img)
        train_set_y_list.append(0)

    for image in (os.listdir(train_clean)): 
        path = os.path.join(train_clean, image)
        img = cv2.imread(path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_set_x_orig_list.append(img)
        train_set_y_list.append(1)

    for image in (os.listdir(test_messy)): 
        path = os.path.join(test_messy, image)
        img = cv2.imread(path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_set_x_orig_list.append(img)
        test_set_y_list.append(0)

    for image in (os.listdir(test_clean)): 
        path = os.path.join(test_clean, image)
        img = cv2.imread(path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_set_x_orig_list.append(img)
        test_set_y_list.append(1)

    train_set_x_orig=np.array(train_set_x_orig_list)
    train_set_y=np.array(train_set_y_list)
    train_set_y=train_set_y.reshape((1,train_set_y.shape[0]))

    test_set_x_orig=np.array(test_set_x_orig_list)
    test_set_y=np.array(test_set_y_list)
    test_set_y=test_set_y.reshape((1,test_set_y.shape[0]))

    classes_list=[b'messy',b'clean']
    classes=np.array(classes_list)
    
    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y,classes=load_dataset()

# To check for a clean room
index = 100
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

# To check for a messy room
index = 10
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

# Find the values for:
# - m_train (number of training examples)
# - m_test (number of test examples)
# - num_px (= height = width of a training image)
# Remember that train_set_x_orig is a numpy-array of shape (m_train, num_px, num_px, 3). For instance, you can access m_train by writing train_set_x_orig.shape[0].

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape (num_px $*$ num_px $*$ 3, 1).
# A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b$*$c$*$d, a) is to use:
# X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], num_px*num_px*3).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
# print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
print('number of train datasets =' + str(train_set_x.shape))
print('number of test datasets =' + str (test_set_x.shape))

print('% of Messy in the training data: ', 100*np.sum(train_set_y == 0)/len(train_set_y[0]))
print('% of Clean in the training data: ', 100*np.sum(train_set_y == 1)/len(train_set_y[0]))

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0.
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    A = A.astype(np.float64)
    cost = -1/m*np.sum(np.nan_to_num(Y*np.log(A)+(1-Y)*np.log(1-A)),axis=1)  
    dw = 1/m*np.dot(X,(A-Y).T)
    db = 1/m*np.sum(A-Y)  
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}    
    return grads, cost

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)  
    costs = []    
    for i in range(num_iterations): 
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w-learning_rate*dw
        b = b-learning_rate*db 
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0,i]<=0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = 0.01, print_cost = True)

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

def ownimage(my_image):
    # We preprocess the image to fit your algorithm.
    fname = my_image
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    plt.imshow(image)
    
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(d["w"], d["b"], image)
    plt.title(classes[int(np.squeeze(my_predicted_image)),].decode("utf-8"))
    result = classes[int(np.squeeze(my_predicted_image)),].decode("utf-8")
    print (result)
    return image, result

    # print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

pickle.dump(ownimage, open("ownimage.pkl", "wb"))

img = "./input/images/test/1.png"
mod = pickle.load(open("ownimage.pkl", "rb"))
image, result = mod(img)
print(result)
# plt.imshow(image)