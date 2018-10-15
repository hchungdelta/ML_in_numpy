'''
Simple neural network in numpy

Architecture : 3 fully connected layers
use square_loss function

Python version : 3.5
Numpy version  : 1.15.1

Date   : 10/15/2018
Author : Hao Chien, Hung.
'''
import numpy as np
import time

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def derv_sigmoid(x):
    return  sigmoid(x)*(1-sigmoid(x))

def sigmoid_xW_b(x,W,b):
    z = np.matmul(x,W) + b
    sigmoid_z = sigmoid(z)
    return sigmoid_z

def square_loss(target,prediction,batch):
    return np.sum (0.5*((target-prediction)**2)  )/batch

def initialize_layer(input_unit,output_unit):
    # generate initial weights and bias
    W=np.random.normal(size=[input_unit,output_unit])
    b=np.random.normal(size=output_unit)
    return W,b

def generate_zeros_like(shape): 
    # similiar to np.zeros_like(), except for many arrays in list
    output_zeros_like=[]
    for a in shape :
        output_zeros_like.append(np.zeros_like(a))
    return output_zeros_like

def onehot(number,depth):
    # e.g. onehot(3,5) --> [0. 0. 0. 1. 0. ]
    array=np.zeros(depth,dtype='f')
    array[number]=1.
    return array


# how many data you want to input 
input_data_amount = 300000

# learning rate
lr = 0.1000 

# how many data you want to train in one step
#  p.s. input_data_amount / batch = total step
batch = 10  

# to print the prediction and 
display_epoch =  1000 

# optional, just to show how much time you cost 
start=time.time()

# generate random weights and bias
W1,b1= initialize_layer(64,32)
W2,b2= initialize_layer(32,16)
W3,b3= initialize_layer(16,10)

total_step  = int(input_data_amount/batch)
total_L=0.  
for a in range(total_step+1):
    # generate input / real output
    my_array=np.random.rand(batch,64)
    target=[]
    for single_data in range(batch) :
        # generate 0 - 10, depends only on the first element in array
        classification= int( 10*my_array[single_data][0]   )    
        target.append( onehot(classification,10)  )

    # Full connected layers
    output1 = sigmoid_xW_b( my_array,W1, b1)
    output2 = sigmoid_xW_b(  output1,W2, b2)
    output3 = sigmoid_xW_b(  output2,W3, b3)

    # Loss function for this batch step (divide by batch)
    L=  square_loss(target,output3,batch)    

    # Loss function for this display epoch (sum up and divide by display_epoch)    
    total_L += L/display_epoch     
       
    if a %  display_epoch==0 and a !=  0 :
        print("step {}, Loss {:.3f}".format(a,total_L)              )
        print("target: {}".format( np.argmax(   target,axis=1 )  )  )  
        print("  pred: {}".format( np.argmax(  output3,axis=1 )  )  )  
        total_L=0

    # generate zeros like np.array as placeholders
    sum_dL3,sum_db3,sum_dW3=generate_zeros_like([b3,b3,W3])    
    sum_dL2,sum_db2,sum_dW2=generate_zeros_like([b2,b2,W2])
    sum_dL1,sum_db1,sum_dW1=generate_zeros_like([b1,b1,W1])

    # backward propagation
    for single_data in range(batch) :
        dLoss=output3[single_data]-target[single_data]
        dL3 = dLoss *derv_sigmoid(output3[single_data])
        db3 = lr *dL3
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
        dW1 = lr *np.outer(  my_array[single_data].T  , dL1     )
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