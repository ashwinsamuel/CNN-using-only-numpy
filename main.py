import numpy as np
import tensorflow as tf

#Convolution layer class
class ConvolutionLayer:
    def __init__(self, kernel_num, kernel_size):

        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.kernels = np.random.randn(kernel_num, kernel_size, kernel_size) / (kernel_size**2) #Dividing by kernel_size^2 for weight normalization 

    def gen(self, image):
 
        image_h, image_w = image.shape
        self.image = image
        for h in range(image_h-self.kernel_size+1):
            for w in range(image_w-self.kernel_size+1):
                patch = image[h:(h+self.kernel_size), w:(w+self.kernel_size)]
                yield patch, h, w
    
    def forward(self, image):

        image_h, image_w = image.shape
        convolution_output = np.zeros((image_h-self.kernel_size+1, image_w-self.kernel_size+1, self.kernel_num))
        for patch, h, w in self.gen(image):
            convolution_output[h,w] = np.sum(patch*self.kernels, axis=(1,2))
        return convolution_output
    
    def back(self, dE_dY, alpha):

        dE_dk = np.zeros(self.kernels.shape) #initializing gradient of the loss function as per the kernel weights
        for patch, h, w in self.gen(self.image):
            for f in range(self.kernel_num):
                dE_dk[f] += patch * dE_dY[h, w, f]
        self.kernels -= alpha*dE_dk
        return dE_dk

#Pooling layer for size compression
class MaxPoolingLayer:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def gen(self, image):

        output_h = image.shape[0] // self.kernel_size
        output_w = image.shape[1] // self.kernel_size
        self.image = image

        for h in range(output_h):
            for w in range(output_w):
                patch = image[(h*self.kernel_size):(h*self.kernel_size+self.kernel_size), (w*self.kernel_size):(w*self.kernel_size+self.kernel_size)]
                yield patch, h, w

    #forward feeding
    def forward(self, image):
        image_h, image_w, num_kernels = image.shape
        max_pooling_output = np.zeros((image_h//self.kernel_size, image_w//self.kernel_size, num_kernels))
        for patch, h, w in self.gen(image):
            max_pooling_output[h,w] = np.amax(patch, axis=(0,1))
        return max_pooling_output

    #back propogation
    def back(self, dE_dY):
        
        dE_dk = np.zeros(self.image.shape)
        for patch,h,w in self.gen(self.image):
            image_h, image_w, num_kernels = patch.shape
            max_val = np.amax(patch, axis=(0,1))

            for idx_h in range(image_h):
                for idx_w in range(image_w):
                    for idx_k in range(num_kernels):
                        if patch[idx_h,idx_w,idx_k] == max_val[idx_k]:
                            dE_dk[h*self.kernel_size+idx_h, w*self.kernel_size+idx_w, idx_k] = dE_dY[h,w,idx_k]
            return dE_dk

class SoftmaxLayer:
    def __init__(self, input_units, output_units):
        self.weight = np.random.randn(input_units, output_units)/input_units
        self.bias = np.zeros(output_units)

    def forward(self, image):
        self.original_shape = image.shape
        image_flattened = image.flatten()
        self.flattened_input = image_flattened
        first_output = np.dot(image_flattened, self.weight) + self.bias
        self.output = first_output
        softmax_output = np.exp(first_output) / np.sum(np.exp(first_output), axis=0)#Activation function
        return softmax_output

    def back(self, dE_dY, alpha):
        for i, gradient in enumerate(dE_dY):
            if gradient == 0:
                continue
            transformation_eq = np.exp(self.output)
            S_total = np.sum(transformation_eq)

            #Calculatiung gradients wrt output
            dY_dZ = -transformation_eq[i]*transformation_eq / (S_total**2)
            dY_dZ[i] = transformation_eq[i]*(S_total - transformation_eq[i]) / (S_total**2)

            # Calculating gradient of output wrt weight and bias
            dZ_dw = self.flattened_input
            dZ_db = 1
            dZ_dX = self.weight

            # Calculating gradient of loss wrt output
            dE_dZ = gradient * dY_dZ

            # Calculating gradient of loss wrt weight and bias
            dE_dw = dZ_dw[np.newaxis].T @ dE_dZ[np.newaxis]
            dE_db = dE_dZ * dZ_db
            dE_dX = dZ_dX @ dE_dZ

            self.weight -= alpha*dE_dw #replacing with new parameters
            self.bias -= alpha*dE_db

            return dE_dX.reshape(self.original_shape)

def CNN_forward(image, label, layers):
    output = image/255.
    for layer in layers:
        output = layer.forward(output)
    #Loss function
    loss = -np.log(output[label])
    accuracy = 1 if np.argmax(output) == label else 0
    return output, loss, accuracy

def CNN_back(gradient, layers, alpha=0.05):
    grad_back = gradient
    for layer in layers[::-1]:
        if type(layer) in [ConvolutionLayer, SoftmaxLayer]:
            grad_back = layer.back(grad_back, alpha)
        elif type(layer) == MaxPoolingLayer:
            grad_back = layer.back(grad_back)
    return grad_back


def CNN_training(image, label, layers, alpha=0.05):
    output, loss, accuracy = CNN_forward(image, label, layers)

 
    gradient = np.zeros(10)
    gradient[label] = -1/output[label]
    gradient_back = CNN_back(gradient, layers, alpha)
    return loss, accuracy

def CNN_testing(image, label, layers, alpha=0.05):
    output, loss, accuracy = CNN_forward(image, label, layers)

    return loss, accuracy

if __name__ == '__main__':
  
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data() # Load training data
  X_train = X_train[:10000]
  y_train = y_train[:10000]
  X_test = X_test[:1000]
  y_test = y_test[:1000]

 # Neural network structure
  layers = [
    ConvolutionLayer(16,3), # layer with 16 3x3 filters, output (26,26,16)
    MaxPoolingLayer(2), # pooling layer 2x2, output (13,13,16)
    SoftmaxLayer(13*13*16, 10) # softmax layer with 13*13*16 input, 10 output
    ] 

  for epoch in range(50):
    print('Epoch {} ->'.format(epoch+1))

    #Training
    train_loss = 0
    train_accuracy = 0
    for i, (image, label) in enumerate(zip(X_train, y_train)):
      if (i+1) % 1000 == 0:
        print("From {} to {} iterations: average loss {}, accuracy {}".format(i-998, i+1, train_loss/1000, train_accuracy/10))
        train_loss = 0
        train_accuracy = 0
        break
      train_loss_1, train_accuracy_1 = CNN_training(image, label, layers)
      train_loss += train_loss_1
      train_accuracy += train_accuracy_1


    #Testing
    test_loss = 0
    test_accuracy = 0
    for i, (image, label) in enumerate(zip(X_test, y_test)):
      test_loss_1, test_accuracy_1 = CNN_testing(image, label, layers)
      test_loss += test_loss_1
      test_accuracy += test_accuracy_1
    print(f"After Epoch {epoch+1} on 1000 test images: average loss {test_loss/len(y_test)}, accuracy {test_accuracy*100/len(y_test)}")