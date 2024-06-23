#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:14:01 2023

"""
"""
Goals: - write an Autoencoder 8 x 3 x 8 units and apply to image compression
       - apply and test Hopfield network capacity
       - emply different dimensionality reduction techniques
"""
import numpy as np
import random
import matplotlib.pyplot as plt

#%% Q1: Self-supervised learning

#References: 
#https://github.com/sje30/dl2023/blob/main/code/xor.R

#define activation functions

def sigmoidal(x):
    """ Sigmoidal activation """
    g = 1 / (1 + np.exp(-x))
    return g

def sigmoidal_prime(x):
    """ Sigmoidal derivative """
    y = sigmoidal(x)
    return y * (1 - y) 


class MLPclassifier:
    def __init__(self, n_inputs, n_hidden):
        
        #initialise parameters
        
      self.n_inputs = n_inputs   #number of input units
      self.n_outputs = n_inputs  #number of hidden units
      self.n_hidden = n_hidden   #number of output units
      
      #iniitalise weight matrices randomly
      self.w1 = np.random.uniform(0, 1, size = (self.n_hidden,(self.n_inputs+1)))
      self.w2 = np.random.uniform(0, 1, size = (self.n_outputs,self.n_hidden+1))
      
      #initialise epoch variables with zeros
      self.y_j =  np.zeros((self.n_hidden+1, 1))
      self.delta_j = np.zeros(self.n_hidden)
      self.delta_k = np.zeros(self.n_outputs)
      
      self.bias = -1 #bias is -1
     
    def feedforward(self, input_nodes):
        
        y_i = np.append(input_nodes, self.bias)   #select the input unit at this iteration
        
        #INPUT TO HIDDEN
        a_j = np.dot(self.w1,y_i)    #matrix multiplication: input X weight matrix 1
        
        y_j = np.append(sigmoidal(a_j), self.bias).reshape(-1,1)
        
        #HIDDEN TO OUTPUT
        a_k = np.dot(self.w2,y_j)      #matrix multiplication: hidden X weight matrix 2
        y_k = sigmoidal(a_k)           #pass these through activation function
        #no need for for loop because there is no bias unit
        
        return a_j, y_j, a_k, y_k, y_i
    
    def compute_error(self, t_k, y_k):
        return np.sum(0.5 * (t_k - y_k.T)**2) #compute error 
    
    def backpropagation(self, a_j, y_j, a_k, y_k, t_k, y_i):
        
        self.delta_k = sigmoidal_prime(a_k).T * (t_k - y_k.T)      #part of the d_w2
        
        for q in range(self.n_hidden+1):
            for r in range(self.n_outputs):
                d_w2[r,q] = d_w2[r,q] + y_j[q] * self.delta_k[:,r] #multiply by input to get d_w2
        
        #Hidden to input
        for q in range(self.n_hidden):
                self.delta_j[q] = sigmoidal_prime(a_j[q]) * np.sum(self.delta_k*self.w2[:,q])
                
        #update delta_j using delta_k and each column of the w2 matrix
        #don't use the last column as that corresponds to bias unit
        #therefore, only loop through J, not J+1
            
        for p in range((self.n_inputs+1)):  #for each column in d-w1
            for q in range(self.n_hidden):  #for each row in d_w1
                d_w1[q,p] = d_w1[q,p] + self.delta_j[q] * y_i[p] 
        
        return d_w1, d_w2

    def weight_update(self, d_w1, d_w2, epsilon = 0.5):
        
       self.w2 = self.w2 + (epsilon*d_w2) #update w2 
       self.w1 = self.w1 + (epsilon*d_w1) #update w1
       
       
#set input nodes
all_x = []
all_activations = []
all_errors = []

for i in range(3):
    x = np.zeros((8,8)) #create a matrix of 8 nodes with 8 units each 
    #select a random index in each node
    activated_nodes = random.sample(range(x.shape[0]),x.shape[1])

    for i in range(x.shape[0]):     #for each node
        x[i,activated_nodes[i]] = 1 #transform the selected unit into 1
        
    all_x += [x]  
    

for i in range(len(all_x)):
    nepochs = 20000              #number of epochs - 1 epoch is the presentation of all weights
    errors = np.zeros(nepochs)   #initialise error convergence var
    inputs = all_x[i]

    model = MLPclassifier(n_inputs = 8, n_hidden = 3)

    for epoch in range(nepochs): 
        
        d_w1 = np.zeros((model.n_hidden,model.n_inputs+1))  #keep a copy of the change in weights
        d_w2 = np.zeros((model.n_outputs,model.n_hidden+1)) #keep a copy of the change in weights
        
        epoch_err = []            #initialise error for the epoch
        
        order = random.sample(range(inputs.shape[0]),inputs.shape[0]) #randomise order of update to avoid bias
        
        for i in range(inputs.shape[0]): #for each row in my data (each has 1 bias unit)
            y_inp = inputs[order[i],:]   #set inputs for epoch
            t_k = inputs[order[i],:] #set target for epoch
            a_j, y_j, a_k, y_k, y_i = model.feedforward(y_inp) #compute feedforward
            epoch_err += [model.compute_error(t_k, y_k)] #compute error
            d_w1, d_w2 = model.backpropagation(a_j, y_j, a_k, y_k, t_k, y_i) #backpropagation
            
        model.weight_update(d_w1, d_w2) #update weights at the end of each epoch
            
        errors[epoch] = np.mean(epoch_err) #compute the mean epoch error

    all_errors += [errors] #add mean epoch error to list
    activations = np.zeros((len(inputs),len(inputs))) #initialise final activations

    for i in range(len(inputs)): 
        activations[i,:] = model.feedforward(inputs[i,:])[3].reshape(-1) #compute final activations

    all_activations += [activations]
    

#plot outputs
    
hfont = {'fontname':'Arial'}  

plt.plot(all_errors[0])
plt.title(f"Final error: {round(all_errors[0][-1],3)}",**hfont,size = 20)
plt.xlabel("Epochs",**hfont,size = 15)
plt.ylabel("Error signal",**hfont,size = 15)

fig, axs = plt.subplot_mosaic("AB;CD;EF",figsize=(8,8))    #get mosaic plot 

n = axs["A"].imshow(all_x[0], cmap = 'tab20c')
axs["A"].set_title("Original",**hfont, size = 20)
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
         size=20, weight='bold')   
fig.colorbar(n, shrink = 0.5) 

o = axs["B"].imshow(all_activations[0], cmap = 'tab20c')
axs["B"].set_title("Output",**hfont, size = 20)
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
          size=20, weight='bold')    
fig.colorbar(o, shrink = 0.5) 

n = axs["C"].imshow(all_x[1], cmap = 'tab20c')
axs["C"].set_title("Original",**hfont, size = 20)
axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
         size=20, weight='bold')   
fig.colorbar(n, shrink = 0.5) 

o = axs["D"].imshow(all_activations[1], cmap = 'tab20c')
axs["D"].set_title("Output",**hfont, size = 20)
axs['D'].text(-0.1, 1.1, 'D', transform=axs['D'].transAxes, 
          size=20, weight='bold')    
fig.colorbar(o, shrink = 0.5) 

n = axs["E"].imshow(all_x[2], cmap = 'tab20c')
axs["E"].set_title("Original",**hfont, size = 20)
axs['E'].text(-0.1, 1.1, 'E', transform=axs['E'].transAxes, 
         size=20, weight='bold')   
fig.colorbar(n, shrink = 0.5) 

o = axs["F"].imshow(all_activations[2], cmap = 'tab20c')
axs["F"].set_title("Output",**hfont, size = 20)
axs['F'].text(-0.1, 1.1, 'F', transform=axs['F'].transAxes, 
          size=20, weight='bold')    
fig.colorbar(o, shrink = 0.5) 

fig.tight_layout()


#%%% IMAGE COMPRESSION

import imageio.v3 as iio
import os
from itertools import islice

os.chdir("/Users/carolinaierardi/Documents/Cambridge/Michaelmas/DeepLearning") #change wd

im = iio.imread('grey_lena.png')
im1 = im[:,:,0]/225

plt.imshow(im1)
plt.title("Original image")

def split_image(image, sub_length):  #function that splits image into subimages
    
    split_horizontal = image.shape[0] / sub_length
    split_vertical = image.shape[1] / sub_length
    
    im_horizontal = np.hsplit(image, split_horizontal)
    im_final = [np.vsplit(i, split_vertical) for i in im_horizontal]
    flatten = [x for xs in im_final for x in xs]
    
    return flatten
    
def reconstruct_image(subimages): #function that reconstructs subimage into one
    
    length = len(subimages)
    section = int(np.sqrt(length))
    
    vertical_cat = []
    for i in range(0, length, section):
         start = i
         stop = i + section
         vertical_cat += [np.concatenate(subimages[start:stop],axis = 0)] 
         
    final_cat = np.concatenate(vertical_cat, axis = 1)    
        
    return np.concatenate(vertical_cat, axis = 1)


sub_images = split_image(im1, 4)  #split subimages

hidden_units = np.append(1,np.arange(2, 12, 2)) #vary number of input units in network
compressed_images = [] 
final_error = []

for h in range(len(hidden_units)): #for each case of hidden units 
    
    nepochs = 200              #number of epochs: 1 epoch is the presentation of all weights
    epoch_len = 10
    errors = np.zeros(nepochs)  #initialise error convergence var
    inputs = sub_images

    compression_model = MLPclassifier(n_inputs = 16, n_hidden = hidden_units[h])

    for epoch in range(nepochs): 
        
        d_w1 = np.zeros((compression_model.n_hidden,compression_model.n_inputs+1)) #keep a copy of the change in weights
        d_w2 = np.zeros((compression_model.n_outputs,compression_model.n_hidden+1)) #keep a copy of the change in weights
        
        epoch_err = []            #initialise error for the epoch
        
        order = random.sample(range(len(inputs)), epoch_len)
        #randomise order of update to avoid bias
        
        for i in range(epoch_len): #for each row in my data (each has 1 bias unit)
            y_inp = inputs[order[i]].reshape(1,-1)
            t_k = y_inp
            a_j, y_j, a_k, y_k, y_i = compression_model.feedforward(y_inp)
            epoch_err += [compression_model.compute_error(t_k, y_k)]
            d_w1, d_w2 = compression_model.backpropagation(a_j, y_j, a_k, y_k, t_k, y_i)
            
        compression_model.weight_update(d_w1, d_w2)
            
        errors[epoch] = np.mean(epoch_err)
    
    final_error += [errors[-1]]
    image_activations = []

    for i in range(len(inputs)): 
        image_activations += [compression_model.feedforward(inputs[i])[3].reshape(4,4)]

    compressed_images += [reconstruct_image(image_activations)]

#make plots 
fig, axs = plt.subplot_mosaic("ABC;DEF",figsize=(12,8))    #get mosaic plot 
fig.suptitle("200 epochs", **hfont, size = 25)

axs["A"].imshow(compressed_images[0])
axs["A"].set_title(f"1 hidden units; MSE {round(final_error[0],2)}",**hfont, size = 20)
axs["A"].set_yticks([])
axs["A"].set_xticks([])

axs["B"].imshow(compressed_images[1])
axs["B"].set_title(f"2 hidden units; MSE {round(final_error[1],2)}",**hfont, size = 20)
axs["B"].set_yticks([])
axs["B"].set_xticks([])

axs["C"].imshow(compressed_images[2])
axs["C"].set_title(f"4 hidden units; MSE {round(final_error[2],2)}",**hfont, size = 20)
axs["C"].set_yticks([])
axs["C"].set_xticks([])

axs["D"].imshow(compressed_images[3])
axs["D"].set_title(f"6 hidden units; MSE {round(final_error[3],2)}",**hfont, size = 20)
axs["D"].set_yticks([])
axs["D"].set_xticks([])

axs["E"].imshow(compressed_images[4])
axs["E"].set_title(f"8 hidden units; MSE {round(final_error[4],2)}",**hfont, size = 20)
axs["E"].set_yticks([])
axs["E"].set_xticks([])

axs["F"].imshow(compressed_images[5])
axs["F"].set_title(f" 10 hidden units; MSE {round(final_error[5],2)}",**hfont, size = 20)
axs["F"].set_yticks([])
axs["F"].set_xticks([])
    #ax.title(f"Final error: {round(title,2)}")


#%% Q2: Hopfield network
#References
#https://www.kaggle.com/code/abirhasan1703100/hopfield-neural-networks-algorithm
#https://www.inference.org.uk/itprnn/book.pdf


def create_patterns(x, p, n):
    
    """
    Create x patterns with each containing p +1s and n -1s
    Return list of patterns
    """
    
    my_list = np.zeros((x, (p+n)))    #initialise list 
    for i in range(x):                #for each pattern  
        my_list[i] = [-1]*n + [1]*p   #assign n -1 and p +1
        random.shuffle(my_list[i])    #randomly shuffle these values  
        
    return(my_list)                   #return full pattern list

def generate_noise(patterns, n):
    
    """
    Function to generate noisy version of original patterns
    """
    noisy = np.copy(patterns)
    
    for i in range(patterns.shape[0]):
        ind = random.sample(range(patterns.shape[1]), n)   #define indices to be switched
        for j in range(n):
            noisy[i,ind[j]] = noisy[i,ind[j]]*(-1) #flip sign of the unit
        
    return(noisy)                                #return noisy patterns


def get_weights(original_patterns):
    
    """
    Obtain weight matrix
    Compute dot product of original patterns and sum
    """
    pattern_size = original_patterns.shape[1]            #define pattern size
    weights = np.zeros((pattern_size, pattern_size))     #initialise weight matrix

    for pattern in original_patterns:                    #iterate over patterns
        pattern = np.reshape(pattern, (1, pattern_size)) #reshape patterns
        weights += np.dot(pattern.T, pattern)            #compute dot product 
        
    np.fill_diagonal(weights, 0)                         #set self-connections to zero
 
    return(weights)                                      #return weight matrix

def hopfield_net(original_patterns, noisy_patterns, weight_mat):
    
    """
    Run Hopfield Network algorithm 
    Takes in original and noisy patterns 
    Returns percentage accuracy for each pattern
    """
    
    pattern_size = len(original_patterns[0])           #obtain size of each pattern


    max_iterations = pattern_size                      #set max iterations to same length as patterns
    percent_correct = np.zeros(len(noisy_patterns))    #empty variable for percent_correct variables

    for i, noisy_pattern in enumerate(noisy_patterns):                 #loop through noisy patterns
        iteration = 0                                                  #initialise iteration variable
        while iteration < max_iterations:                              #run until reaches max iterations
            iteration += 1                                             #bookeeping of iteration variable
            prev_pattern = np.copy(noisy_pattern)                      #make copy of noisy pattern
            noisy_pattern = np.sign(np.dot(noisy_pattern, weight_mat)) #apply sign function
            if np.array_equal(noisy_pattern, prev_pattern):            #stop once two consecutive
                                                                       #patterns are the same
                break
                
        percent_correct[i] = (len(np.where(original_patterns[i] == noisy_pattern)[0])/pattern_size)*100
        
        # print the original pattern, noisy pattern, and recovered pattern
        print('{:<20} {:<10} {:<10}'.format("Original pattern", i, str(tuple(original_patterns[i]))))
        print('{:<20} {:<10} {:<10}'.format("Noisy pattern", i, str(tuple(noisy_patterns[i]))))
        print('{:<20} {:<10} {:<10}'.format("Recovered pattern", i, str(tuple(noisy_pattern))))
        print(percent_correct[i])
        print("")
        
    
    return(percent_correct)


def full_pipeline(x, p, n, noise_n, lose_weights = False, lose_n = None):
    
    """
    Parameters
    ----------
    x : integer
        Number of patterns to create.
    p : integer
        Number of +1 in each pattern.
    n : integer
        Number of -1 in each pattern.

    Returns
    -------
    Sparseness of patterns, pattern size, 
    number of patterns and accuracy vector.

    """
    if lose_weights == True:
        initial_pat = create_patterns(x, p, n)                   #create patterns
        noisy_pat = generate_noise(initial_pat, noise_n)         #add noise to them 
        weights = get_weights(initial_pat)                       #obtain weight matrix
        
        samp = random.choices(range(weights.shape[0]), k = 2*lose_n) #sample of indices to have weights lost
        new_weights = np.copy(weights)                               #make copy of weight matrix
        
        for i in np.arange(0, 2*lose_n, 2):           #iterate only over even indices
            new_weights[samp[i], samp[i+1]] = 0       #assign indices to be 0   
        
        accuracy = hopfield_net(initial_pat, noisy_pat, new_weights) #apply hopfield algorithm
    else:
        initial_pat = create_patterns(x, p, n)                   #create patterns
        noisy_pat = generate_noise(initial_pat, noise_n)         #add noise to them 
        weights = get_weights(initial_pat)                       #obtain weight matrix
        accuracy = hopfield_net(initial_pat, noisy_pat, weights) #apply hopfield algorithm
        
    
    return(accuracy)


#%% Noise

rep = 100                                                       #number of repetitions
avg_accuracy_noise = []                                         #empty list for mean accuracies 


noise_array = np.arange(1,100, 5)                               #pattern sizes to be investigated

for j in range(len(noise_array)):                               #for each pattern size
    accuracy = []                                               #empty list to store accuracy vectors
    for i in range(rep): #repetitions
        accuracy += [full_pipeline(10, 50, 50, noise_array[j])] #perform full pipeline
    avg_accuracy_noise += [np.mean(accuracy)]                   #compute mean accuracy
      
plt.plot(noise_array/100, avg_accuracy_noise, color = '#3293a8')#plot pattern size and acc

hfont = {'fontname':'Arial'}                                    #new font
plt.title("Hopfield network noise",**hfont, fontsize = 20)      #plot title
plt.xlabel("Noise",**hfont, fontsize = 15);                     #x-axis label
plt.ylabel("Accuracy",**hfont, fontsize = 15);                  #y-axis label


#%% Capacity
rep = 100                                                     #number of repetitions
avg_accuracy_capacity = []                                    #empty list for mean accuracies 
noise = 25


pattern_sizes = np.arange(start = 3,stop = 31)                #pattern sizes to be investigated

for j in range(len(pattern_sizes)):                           #for each pattern size
    accuracy = []                                             #empty list to store accuracy vectors
    for i in range(rep):                                      #repetitions
        accuracy += [full_pipeline(pattern_sizes[j], 50, 50, noise)] #perform full pipeline
    avg_accuracy_capacity += [np.mean(accuracy)]                     #compute mean accuracy
      

plt.plot(pattern_sizes/100, avg_accuracy_capacity, color = '#3293a8')    #plot pattern size and acc
plt.title("Hopfield network capacity",**hfont, fontsize = 20)   #plot title
plt.xlabel("Pattern size/neuron",**hfont, fontsize = 15);       #x-axis label
plt.ylabel("Accuracy",**hfont, fontsize = 15);                  #y-axis label

#%% Sparcity 

x = 12                                              #fix pattern size
p_array = np.arange(5, 100, 5)                      #set values to put +1s to
n = 100 - p_array                                   #we want each pattern to have 100 neurons, 
                                                    #so substract from +1s to get -1s

avg_accuracy_sparcity = []                          #empty list for mean accuracies 

for j in range(len(p_array)):                       #for each value of p
    accuracy = []                                   #empty list for accuracy vectors
    for i in range(rep):                            #for all repetitions
        accuracy += [full_pipeline(x, p_array[j], n[j], noise)]  #perform pipeline
    avg_accuracy_sparcity += [np.mean(accuracy)]    #compute mean accuracy 
      

# plt.plot(p/100, avg_accuracy_sparcity, color = '#3293a8')     #plot +1 proportion and acc

# hfont = {'fontname':'Arial'}                                  #new font
# plt.title("Hopfield network sparcity",**hfont, fontsize = 20) #plot title
# plt.xlabel("+1 proportion",**hfont, fontsize = 15);           #x-axis label
# plt.ylabel("Accuracy",**hfont, fontsize = 15);                #y-axis label


#%% Robustness

weight_loss_numbers = np.arange(0, 10000, 50)
avg_acc_robust = []
x = 12
p = 50
n = 50


for j in range(len(weight_loss_numbers)):                     #for each pattern size
    accuracy = []                                             #empty list to store accuracy vectors
    for i in range(rep): #repetitions
        accuracy += [full_pipeline(x, p, n, noise,
                                   lose_weights=True, lose_n = weight_loss_numbers[j])] #perform full pipeline
    avg_acc_robust += [np.mean(accuracy)]                       #compute mean accuracy
      

# plt.plot(weight_loss_numbers, avg_acc_robust, color = '#3293a8') #plot weights lost and acc

# hfont = {'fontname':'Arial'}                                    #new font
# plt.title("Hopfield network robustness",**hfont, fontsize = 20) #plot title
# plt.xlabel("Weights lost",**hfont, fontsize = 15);              #x-axis label
# plt.ylabel("Accuracy",**hfont, fontsize = 15);                  #y-axis label


#%% Different method to set weights


def sigmoid(x):
    """
    Function to perform sigmoid activation
    """
    return 1 / (1 + np.exp(-x))

def mod_weights(patterns, eta, alpha):
    """
    patterns : p X n matrix
        Input patterns.
    eta : float
        value for eta.
    alpha : float
        value for alpha.

    Returns: Weight matrix.

    """
    
    t = np.copy(patterns)             #make copy of input patterns
    inds = np.where(patterns == -1)   #find where values are -1

    for i in range(len(inds[0])):
        t[inds[0][i],inds[1][i]] = 0  #transform these alues into 0
    
    w = get_weights(patterns)        #obtain original Hebbian weights   

    L = 10                           #perform 10 times

    for _ in range(L):

        a = np.dot(patterns, w)      #compute all activations
        y = sigmoid(a)               #compute all outputs
        e = t - y                    #compute all errors
        gw = np.dot(patterns.T, e)   #compute the gradients
        gw = gw + gw.T               #symmetrize gradients
        w = w + eta * (gw - alpha * w)     # make step
        
    return(w)
    
def mod_pipeline(rep, eta, alpha):
    
    avg_accuracy = []                                       #empty list to store mean accuracy
    accuracy = []                                           #empty list to store accuracy vectors
    for i in range(rep):                                    #repetitions
        patterns = create_patterns(10, 50, 50)              #create input patterns
        noise = generate_noise(patterns, 25)                #generate noisy patterns
        weights = mod_weights(patterns, eta, alpha)         #obtain weight matrix
        accuracy = [hopfield_net(patterns, noise, weights)] #perform full pipeline
        avg_accuracy += [np.mean(accuracy)]                 #compute mean accuracy

    return(avg_accuracy)
      

accuracy_final = []
accuracy_final_mean = []


for i in range(1000):                                  #for all repetitions
    accuracy_final += [full_pipeline(10, 50, 50, 25)]  #perform pipeline
    accuracy_final_mean += [np.mean(accuracy_final)]   #average accuracy

  
etas = np.arange(1,11,1)                #range of etas values to test
alphas = np.arange(0.001, 1, 0.1)       #range of alpha values to test
accuracy_param = []                     #empty list for accuracy
    
for a in alphas:                        #loop over alphas 
    for e in etas:                      #loop over etas
        acc = mod_pipeline(100, e, a)   #perform 100 iterations with each value
        accuracy_param += [np.mean(acc)]#compute average mean
        print(a,e,accuracy_param)       #print parameters and average mean
        
alpha = 0.2         #chosen alpha
eta = 1             #chosen eta

acc = mod_pipeline(1000, alpha, eta)    #perform pipeline with chosen parameters


#%% make figures
hfont = {'fontname':'Helvetica'}        #new font

fig, (ax1, ax2) = plt.subplots(1, 2) 
                                    #plot with subplots
fig.suptitle("Hopfield network performance", **hfont, size = 20)         #title for whole figure
ax1.hist(accuracy_final_mean, color = '#3293a8',ec = "black", bins = 30) #plot weights lost and acc
ax1.set_title("Hebbian weights",**hfont, fontsize = 15)                  #plot title
ax1.set_xlabel("Accuracy",**hfont, fontsize = 15);                       #x-axis label
ax1.set_ylabel("Frequency",**hfont, fontsize = 15);                      #y-axis label

ax2.hist(acc, color = '#3293a8',ec = "black", bins = 30)                 #plot weights lost and acc
ax2.set_title("Perceptron weights",**hfont, fontsize = 15)               #plot title
ax2.set_xlabel("Accuracy",**hfont, fontsize = 15);                       #x-axis label
ax2.set_ylabel("Frequency",**hfont, fontsize = 15);                      #y-axis label

fig.tight_layout()

fig, axs = plt.subplot_mosaic("AB;CD",figsize=(10,6))                    #get mosaic plot 
fig.tight_layout(h_pad = 2)                                              #tight layout so there is no overlay between plots

axs["A"].plot(noise_array/100, avg_accuracy_noise, color = '#3293a8')    #plot pattern size and acc
axs["A"].set_title("Hopfield network noise",**hfont, fontsize = 20)   #plot title
axs["A"].set_xlabel("Noise",**hfont, fontsize = 15);                     #x-axis label
axs["A"].set_ylabel("Accuracy",**hfont, fontsize = 15);                  #y-axis label
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=20, weight='bold')   

axs["B"].plot(pattern_sizes/100, avg_accuracy_capacity, color = '#3293a8')#plot pattern size and acc
axs["B"].set_title("Hopfield network capacity",**hfont, fontsize = 20)        #plot title
axs["B"].set_xlabel("Pattern size/neurons",**hfont, fontsize = 15);            #x-axis label
axs["B"].set_ylabel("Accuracy",**hfont, fontsize = 15);                       #y-axis label
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=20, weight='bold')

axs["C"].plot(p_array/100, avg_accuracy_sparcity, color = '#3293a8')     #plot +1 proportion and acc
axs["C"].set_title("Hopfield network sparsity",**hfont, fontsize = 20) #plot title
axs["C"].set_xlabel("+1 proportion",**hfont, fontsize = 15);           #x-axis label
axs["C"].set_ylabel("Accuracy",**hfont, fontsize = 15);                #y-axis label
axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
            size=20, weight='bold')   

axs["D"].plot(weight_loss_numbers, avg_acc_robust, color = '#3293a8') #plot weights lost and acc
axs["D"].set_title("Hopfield network robustness",**hfont, fontsize = 20) #plot title
axs["D"].set_xlabel("Weights lost",**hfont, fontsize = 15);              #x-axis label
axs["D"].set_ylabel("Accuracy",**hfont, fontsize = 15);                  #y-axis label
axs['D'].text(-0.1, 1.1, 'D', transform=axs['D'].transAxes, 
            size=20, weight='bold')   
fig.tight_layout(h_pad = 2)                                              #tight layout so there is no overlay between plots


#%% Dimensionality reduction

#Code adapted from
#https://builtin.com/data-science/tsne-python

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

import seaborn as sns

cov_type = fetch_olivetti_faces()

X = cov_type.data
y = cov_type.target

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)
df['label'] = df['y'].apply(lambda i: str(i))

np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

plt.gray()
fig = plt.figure( figsize=(16,10) )
fig.tight_layout(h_pad = 2)

for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1, title="Face: {}".format(str(df.loc[rndperm[i],'label'])) )
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((64,64)).astype(float))
plt.show()

#%%% PCA
pca = PCA(n_components=10)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[feat_cols].values)
pca_result = pca.fit_transform(data_scaled)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 

plt.plot(pca.explained_variance_ratio_, linestyle = 'dashed');             #obtain a plot for the explained variance for the components 
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("N components")                                 # x label
plt.ylabel("Explained variance")                           # y label
plt.title("Variance explained ratio for PCA")              # plot title

print(np.sum(pca.explained_variance_ratio_))               # visually, the first component explains a lot of the variance, so let's get a quantitative score of this         


#%%% t-SNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df[feat_cols].values)

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

#%%% UMAP
import umap
import umap.plot


mapper = umap.UMAP().fit(cov_type.data)

umap.plot.points(mapper, labels=cov_type.target, width = 800, height=800)

#%%% Figures

fig, axs = plt.subplot_mosaic("ABC",figsize=(20,10))                    #get mosaic plot 
fig.tight_layout(h_pad = 2)                                              #tight layout so there is no overlay between plots

sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("husl", 40),
    data=df.loc[rndperm,:],
    legend="full", ax = axs["A"]
)
axs["A"].set_title("PCA, components = 2, cum. var. = 0.39", size = 20)
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=20, weight='bold')   

sns.scatterplot( 
     x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 40),
    data= df,
    legend="full", ax = axs["B"]
);
axs["B"].set_title("t-SNE, perplexity = 50", size = 20);
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=20, weight='bold')


umap.plot.points(mapper, labels=cov_type.target, width = 1000, height=1000, cmap = 'hsl', ax = axs["C"])
axs["C"].set_title("UMAP, N_neighbours = 20", size = 20)
axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
            size=20, weight='bold')   


from io import StringIO
from scipy.io import mmread
os.chdir("/Users/carolinaierardi/Downloads")
m = mmread("matrix.mtx.gz")

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, init = "random", method = "exact")
tsne_results = tsne.fitm(m)

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
