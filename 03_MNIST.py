import numpy as np
import time
import MNIST_data_loader as MNIST_load

# input data : (,784)  28*28 pixels
# target data :(,10)  one hot already
training_data, validation_data, test_data = MNIST_load.load_data()


def sigmoid(x):
    return (1/(1+np.exp(-x)))
def derv_sigmoid(x):
    return  sigmoid(x)*(1-sigmoid(x))
def sigmoid_xW_b(x,W,b):
    z = np.matmul(x,W) + b
    sigmoid_z = sigmoid(z)
    return sigmoid_z
def xW_b(x,W,b):
    z = np.matmul(x,W) + b
    return z

def initialize_layer(input_unit,output_unit):
    W=np.random.normal(size=[input_unit,output_unit])
    b=np.random.normal(size=output_unit)
    return W,b
def generate_zeros_like(shape):
    output_zeros_like=[]
    for a in shape :
        output_zeros_like.append(np.zeros_like(a))
    return output_zeros_like
def cross_entropy(target,prediction,batch):
    small_num =np.zeros_like(target)
    small_num.fill(1e-8)   # prevent log(0)
    return  -np.sum(np.multiply(target,np.log(prediction+small_num) )  )/batch
def softmax(x):
    after_softmax=[]
    for row in range(x.shape[0]) :
        this_row=np.exp(x[row])/np.sum(np.exp(x[row]))
        after_softmax.append(this_row)
    return np.array(after_softmax)
def accuracy_test(pred,target):
    accuracy = 0 
    for element in range(len(pred)):
        if pred[element] == target[element] :
            accuracy += 1./len(pred)
    return accuracy
def onehot(number,depth):
    array=np.zeros(depth,dtype='f')
    array[number]=1.
    return array
 
start=time.time()

#learning rate
lr = 0.15
batch = 32 
training_data_amount = len(training_data)  # 50000 for MNIST training data

# total_step_in_one_loop = 50000 / batch 
total_step_in_one_loop = int(training_data_amount/batch )

# How many loops you want to go through
amount_of_loop = 10

#  display the prediction and target after finishing every 20% training data
display_epoch = int(  0.2*total_step_in_one_loop   )

# using validation data to esimate the accuracy (when finishing a loop)
use_validation = True
display_epoch_for_validation = total_step_in_one_loop



W1,b1= initialize_layer(784,100)
W2,b2= initialize_layer(100,30)
W3,b3= initialize_layer(30,10)


total_L=0.
offset=0
for a in range(amount_of_loop*total_step_in_one_loop+1):
    # input MNIST data
    target=[]
    training_batch=[]
    if offset+batch > training_data_amount :
        offset=0  
    for single_data in range(offset,offset+batch  ) :
        _reshape_input=training_data[single_data][0] 
        training_batch.append(_reshape_input)  
        _reshape_target=training_data[single_data][1] 
        target.append(_reshape_target)
    offset += batch
    # arrayize 
    training_batch=np.array(training_batch)
    target= np.array(target)

    # Full connected layers
    output1 = sigmoid_xW_b( training_batch,W1, b1)
    output2 = sigmoid_xW_b(  output1,W2, b2)
    output3 = xW_b(  output2,W3, b3)
    pred    = softmax(output3)

    # Loss function
    L= cross_entropy(target,pred,batch)
    total_L += L/display_epoch

    # use training data to estimate the accuracy
    if a %  display_epoch==0   and a != 0 : 
        TARGET =np.argmax(target,axis=1 )
        PREDICT=np.argmax(pred,axis=1   )
        accur=accuracy_test( TARGET   , PREDICT )
        print("step {}, Loss {:.3f}".format(a,total_L)            )
        print("target: {}".format( TARGET  )                      )  
        print("  pred: {}".format( PREDICT )                      )
        print("Accuracy of trainin data : {:.3f}".format(accur)   )
        total_L=0

    # using validation data to estimate the accuracy
    if  a %  display_epoch_for_validation ==0  and a !=0 and use_validation == True:
        validation_target=[]
        validation_batch =[]
        for single_data in range(np.array(validation_data).shape[0]) :
            validation_batch.append(  np.array(validation_data[single_data][0])) 
            validation_target.append( np.array(validation_data[single_data][1])) 

        # arrayize
        validation_batch = np.array(validation_batch)
        validation_target= np.array(validation_target)
        # Full connected layers
        output1 = sigmoid_xW_b( validation_batch,  W1, b1)
        output2 = sigmoid_xW_b(          output1,  W2, b2)
        output3 =         xW_b(          output2,  W3, b3)
        pred    =      softmax(output3)

        TARGET  =np.argmax(validation_target,axis=1 )
        PREDICT =np.argmax(pred,axis=1   )
        accuracy=accuracy_test(  TARGET , PREDICT) 
        print("Validation target: {}".format( TARGET   )               )
        print("Validation output: {}".format( PREDICT  )               )
        print("Accuracy of validation data : {:.3f}".format(accuracy)  )

    # generate zeros like np.array as placeholders
    sum_dL3,sum_db3,sum_dW3=generate_zeros_like([b3,b3,W3])    
    sum_dL2,sum_db2,sum_dW2=generate_zeros_like([b2,b2,W2])
    sum_dL1,sum_db1,sum_dW1=generate_zeros_like([b1,b1,W1])

    # backward propagation
    for single_data in range(batch) :

        dLoss = -target[single_data] +pred[single_data]
        dL3 = dLoss
        db3 = lr  *dL3
        dW3 = lr *np.outer( output2[single_data].T,dL3)
        sum_dL3=np.add(sum_dL3,dL3) 
        sum_db3=np.add(sum_db3,db3)
        sum_dW3=np.add(sum_dW3,dW3)

        dL2  = np.multiply( np.matmul(dL3,W3.T) ,derv_sigmoid(output2[single_data]) )
        db2 = lr *dL2
        dW2 = lr *np.outer(  output1[single_data].T  , dL2     )    
        sum_dL2=np.add(sum_dL2,dL2)
        sum_db2=np.add(sum_db2,db2)
        sum_dW2=np.add(sum_dW2,dW2)

        dL1  = np.multiply( np.matmul(dL2,W2.T) ,derv_sigmoid(output1[single_data]) )    
        db1 = lr *dL1
        dW1 = lr *np.outer(  training_batch[single_data].T  , dL1     )
        sum_dL1=np.add(sum_dL1,dL1)
        sum_db1=np.add(sum_db1,db1)
        sum_dW1=np.add(sum_dW1,dW1)

    # New weights and bias
    b3 = b3 - (1./batch)*sum_db3
    W3 = W3 - (1./batch)*sum_dW3
    b2 = b2 - (1./batch)*sum_db2
    W2 = W2 - (1./batch)*sum_dW2
    b1 = b1 - (1./batch)*sum_db1
    W1 = W1 - (1./batch)*sum_dW1


print("COST:{:.3f} seconds".format(time.time() - start) )

