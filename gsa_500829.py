#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:40:18 2024


"""

import os                               #directory changing
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines



os.chdir("~/Documents/Cambridge/Michaelmas/GenomeSequenceAnalysis") #change wd


#%% Q1: Implement a HMM

#Download files in .txt format
# The following code is largely based on the implementation given in: 
    #https://www.vle.cam.ac.uk/pluginfile.php/27976427/mod_resource/content/1/hmm-code-handout.pdf
    #the handout of HMM implementation found on moodle 

#Download files
A = np.loadtxt("A.txt")
B = np.loadtxt("B.txt")
V = np.loadtxt("V.txt")
S = np.loadtxt("S.txt")
mu_0 = np.loadtxt("mu_not.txt")

#make into dataframes and name rows and columns
dfA = pd.DataFrame(A, index = S, columns=S)
df_mu0 = pd.DataFrame(mu_0, index = S)
dfB = pd.DataFrame(B, index = S, columns = V)


def markov_chain(trans_mat, initial_dis, length): 
    
    """
    Samples a markov chain given a transition matrix trans_mat,
    an initial distribution inital_dis and a length "length"
    """
    S = list(trans_mat.index)                                       #obtain state space
    chain = [0] * length                                            #initialise vector
    chain[0] = np.random.choice(S, 1, p=initial_dis.values.T[0])[0] #random initial state
        
    for im in range(1, length):                                     #for each time step, select 
        chain[im] = np.random.choice(S, 1, p=trans_mat.loc[chain[im - 1]].values)[0] #a state space
         
    return chain                                                    #return a markov chain


def hmm(trans_mat, initial_dis, pemit, length):
    
    """
    Simulate an observed sequence from a HMM with transition matrix trans_mat, 
    initial distribution initial_dis, emission matrix pemit nad length "length"
    """

    hidden = markov_chain(trans_mat, initial_dis, length) #obtain markov chain

    emit_states = list(pemit.columns)                     #obtain emission states
    emitted = [np.random.choice(emit_states, 1, p=pemit.loc[x].values)[0] for x in hidden]
    #select an emission state given a probability of the hidden states


    return {"hid": hidden, "emit": emitted}               #return emission and hidden observations


my_len = 115                        #for 115 timesteps
my_hmm = hmm(dfA,df_mu0,dfB,my_len) #compute HMM

#set up legend blobs
lab1 = mlines.Line2D([], [], color="#1B2352", marker='.',
                          markersize=15, label='0')
lab2 = mlines.Line2D([], [], color="#F1C7F0", marker='.',
                          markersize=15, label='1')

#make plot of emission states color-coded with the hidden states
plt.scatter(range(115), my_hmm['emit'], c=my_hmm['hid'], cmap = 'tab20b', alpha = 0.7)
plt.legend(handles = [lab1, lab2], title = "Hidden States", loc = 0) #plot legend
plt.title("Discrete HMM implementation")       #plot title
plt.ylabel("Emission and hidden state spaces") #plot y-label

plt.savefig('Q1.png',bbox_inches='tight')      #save figure
np.savetxt('emission.txt',  my_hmm['emit'])    #save emission state for Q2

#%% Q2: implement forward algorithm


emissions = np.loadtxt("emission.txt") #load emission states

    
def forward(V, a, b, initial_distribution):
    
    alpha = np.zeros((V.shape[0], a.shape[0])) #initialise alpha
    unscaled_alpha = np.zeros((V.shape[0], a.shape[0])) #unscaled alpha
    scaling = np.zeros(V.shape[0]) #scaling factors
    
    #Initialisation - first set of alphas
    unscaled_alpha[0, :] = initial_distribution * b[:, int(V[0])]
    scaling[0] = np.sum(initial_distribution * b[:, int(V[0])])
    alpha[0,:] = unscaled_alpha[0, :] /scaling[0]
    
    #Recursion
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            
            #use formula to adapt subsequent alphas
            unscaled_alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, int(V[t])]
        scaling[t] = np.sum(unscaled_alpha[t,:])
        alpha[t, :] = unscaled_alpha[t,:] / scaling[t]
 
    return alpha, np.sum(np.log(scaling)) #return alpha matrix and log-likelihood
         

ind_emissions = emissions - 1 #for indexing

fw_alpha, fw_like = forward(ind_emissions,A, B, mu_0) #compute matrix and likelihood
print(fw_like) #print output


#%% Q3: log likelihood of yeast GC content

#the sequence should be downloaded in a fasta file
infile = open('Saccharomyces_cerevisiae.R64-1-1.dna.chromosome.III.fa')
chr3 = []               #empty file for sequencing
for line in infile:
    chr3.append(line)   #append line in file to list


chr3_filt = chr3[1:]                                     #exclude heading
chr3_nosp = []                                           #new empty file to remove spaces
for lines in chr3_filt:                                  #for each line in file
    chr3_nosp += [lines.strip()]                         #remove the spaces
    
chr3 = [num for sublist in chr3_nosp for num in sublist] #flatten list


def split_seq(sequence, char):
    
    newseq = [sequence[i:i+char] for i in range(0, len(sequence), char)]
    string = ""                                              #initialise empty string
    newseq_binned = []                                       #initialise new list
    for line in newseq:                                      #for each element in the old list
        newseq_binned += [string.join(line)]                 #join the characters

    return(newseq_binned)

n = 100                                                  #bin at every 100 bp
chr3 = split_seq(chr3, n)                                #separate at 100 bp


def calc_GC(list_sequence):
    
    GCcontent = []                                         #calculate GC content

    for line in list_sequence:                             #for each line in list
        GCcontent += [line.count("C") + line.count("G")]   #add Gs and Cs
    return(GCcontent)

myGC = calc_GC(chr3)

#https://stackoverflow.com/questions/67838417/
#assign-category-or-integer-to-row-if-value-within-an-interval

bounds = np.arange(np.min(myGC), np.max(myGC) + 12, 12)    #make binning scheme
dist_bins = pd.cut(myGC, bins=bounds, include_lowest=True) #use pd.cut to arrange values into bins
res = (dist_bins._ndarray + 1)                             #obtain new categories

new_emissions = res.astype(np.float64)                     #make into float to match old emissions

new_emissions = new_emissions - 1

fw_alphaGC, fw_likeGC = forward(new_emissions, A,B, mu_0)
print(fw_likeGC)

fig, axs = plt.subplot_mosaic("AB",figsize=(10,6))  #get mosaic plot 
hfont = {'fontname':'Arial'}                        #set font

axs["A"].hist(myGC,density=True, edgecolor='black', linewidth=1, color ='#48D1CC') #make plot
axs["A"].set_title("GC content of yeast chromosome III",**hfont, size = 20)        #plot title
axs["A"].set_ylabel("Frequency",**hfont,size = 15)                                 #y-label
axs["A"].set_xlabel("GC content",**hfont,size = 15)                                #x-label
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes,  
         size=20, weight='bold')                                                   #plot reference

axs["B"].hist(new_emissions,density=True, edgecolor='black', linewidth=1, color ='#48D1CC')
axs["B"].set_title("New emission distribution for HMM",**hfont,size = 20)  #plot title
axs["B"].set_ylabel("Frequency",**hfont,size = 15)                         #y-label
axs["B"].set_xlabel("Emission state space",**hfont,size = 15)              #x-label
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
         size=20, weight='bold')                                           #plot ref
fig.tight_layout(h_pad = 1)

plt.savefig('Q3.png',bbox_inches='tight')


#%% Q4: Baum-Welch to estimate model parameters

         
def backward(V, a, b):
    #similar to forward algorithm
    beta = np.zeros((V.shape[0], a.shape[0]))
    unscaled_beta = np.zeros((V.shape[0], a.shape[0]))
    scaling = np.zeros(V.shape[0])
 
    # setting beta(T) = 1
    unscaled_beta[V.shape[0] - 1] = np.ones((a.shape[0]))
    scaling[0] = np.sum(unscaled_beta[V.shape[0] - 1])
    beta[V.shape[0] - 1] = unscaled_beta[V.shape[0] - 1] /scaling[0]
 
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            unscaled_beta[t, j] = (beta[t + 1] * b[:, int(V[t + 1])]).dot(a[j, :])
        scaling[t] = np.sum(unscaled_beta[t,:])
        beta[t, :] = unscaled_beta[t,:] / scaling[t]
 
    return beta
 
 
def baum_welch(V, a, b, initial_distribution, n_iter=100, threshold = 0.001):
    M = a.shape[0]  #state space 
    T = len(V)      #emission variables
    likelihood = [] #initialise likelihood convergence
 
    for n in range(n_iter): #for every iteration 
        
        alpha, like = forward(V, a, b, initial_distribution) #compute forward algorithm
        likelihood.append(like) #add likelihood to convergence
        beta = backward(V, a, b) #compute backward
 
        xi = np.zeros((M, M, T - 1)) #initialise xi 
        for t in range(T - 1): #for each timepoint
            #set the scaler
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, int(V[t + 1])].T, beta[t + 1, :])
            for i in range(M):
                #and then compute xi 
                numerator = alpha[t, i] * a[i, :] * b[:, int(V[t + 1])].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator #make sure it is scaled 
 
        gamma = np.sum(xi, axis=1) #compute gamma
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1)) #update A matrix
 
        # add timepoint in gamma for calculations
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        
        initial_distribution = gamma[:,0]   #set initial distribution
        
        K = b.shape[1]                      #for each element in the state space 
        denominator = np.sum(gamma, axis=1) #set the denominator
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1) #update B matrix
 
        b = np.divide(b, denominator.reshape((-1, 1))) #make sure it is scaled
        
        #when threshold is hit, stop algorithm
        if len(likelihood) > 2 and abs(likelihood[-1] - likelihood[-2]) < threshold: 
            break
 
    return {"a":a, "b":b, "mu":initial_distribution, "likelihood":likelihood}

new_param  = baum_welch(new_emissions, A, B, mu_0, n_iter=40) #

plt.plot(new_param["likelihood"], color = '#4BC5D4')
plt.title(f"Log-likelihood = {round(np.max(new_param['likelihood']),2)}")
plt.xlabel("Iterations")
plt.ylabel("Log-likelihood")


#%% Q5: infer the likely sequence of hidden states

#perform Viterbi algorithm

#https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm


# def viterbi(oy, oA, oB, oPi):
 
#     Pi = oPi.values
#     B = oB.values
#     A = oA.values
#     y = [int(i) for i in oy]
#     T = len(y)
    
#     # Cardinality of the state space
#     K = A.shape[0]
#     T1 = np.empty((K, T), 'd')
#     T2 = np.empty((K, T), 'B')
 


#     # Initilaize the tracking tables from first observation
#     T1[:, 0] = np.log(Pi.T) + np.log(B[:, y[0]-1])
#     T2[:, 0] = 0

#     # Iterate throught the observations updating the tracking tables
#     for i in range(1, T):
#         T1[:, i] = np.max(T1[:, i - 1] + np.log(A.T) + np.log(B[np.newaxis, :, y[i]-1].T), 1)
#         T2[:, i] = np.argmax(T1[:, i - 1] + np.log(A.T), 1)

#     # Build the output, optimal model trajectory
#     x = np.empty(T, 'B')
#     x[-1] = np.argmax(T1[:, T - 1])
#     for i in reversed(range(1, T)):
#         x[i - 1] = T2[x[i], i]

#     return x

#x1 = viterbi(new_emissions, new_params["a"], new_params["b"], new_params["mu"])

#alpha matrix with new parameters
alpha, like = forward(new_emissions, new_param["a"], new_param["b"], new_param["mu"])
#beta matrix with new parameters
beta = backward(new_emissions, new_param["a"], new_param["b"])

denominator = np.sum(alpha*beta, 1)                         #scaler
posterior = np.zeros((alpha.shape))                         #initialise posterior dist
for i in range(len(denominator)):
    posterior[i,:] = alpha[i,:] *beta [i,:]/denominator[i] #set posterior distribution
    
    
plt.scatter(range(len(new_emissions[1800:])), new_emissions[1800:], 
            c=np.argmax(posterior,1)[1800:], cmap = 'tab20b', alpha = 0.7)
plt.legend(handles = [lab1, lab2], title = "Hidden States", loc = 0) #plot legend
plt.title("HMM for GC content of yeast chromosome III")              #plot title
plt.ylabel("Emission and hidden state spaces")                       #plot y-label

plt.savefig('Q5.png',bbox_inches='tight')                            #save figure


#%% Q6: Select different emission space and emission distribution

#maybe a Gaussian distribution ak a continuous emission state space
#we then apply baum welch algortihm and the viterbi algorithm for the new model


from scipy.stats import norm

def Gauss_forward(V, a, my_mean, my_cov, initial_distribution):
    
    #replace b-matrix with drawing from normal distribution
    
    alpha = np.zeros((V.shape[0], a.shape[0]))
    unscaled_alpha = np.zeros((V.shape[0], a.shape[0]))
    scaling = np.zeros(V.shape[0])
    
    unscaled_alpha[0, :] = initial_distribution * norm.pdf(V[0], loc=my_mean, scale=np.sqrt(my_cov))
    scaling[0] = np.sum(initial_distribution * norm.pdf(V[0], loc=my_mean, scale=np.sqrt(my_cov)))
    alpha[0,:] = unscaled_alpha[0, :] /scaling[0]
 
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            
            unscaled_alpha[t, j] = alpha[t - 1].dot(a[:, j]) * norm.pdf(V[t-1], 
                                    loc=my_mean[j], scale=np.sqrt(my_cov[j]))
        scaling[t] = np.sum(unscaled_alpha[t,:])
        alpha[t, :] = unscaled_alpha[t,:] / scaling[t]
 
    return alpha, np.sum(np.log(scaling))

         
def Gauss_backward(V, a, my_mean, my_cov):
    #replace b-matrix with drawing numbers from normal distribution 
    beta = np.zeros((V.shape[0], a.shape[0]))
    unscaled_beta = np.zeros((V.shape[0], a.shape[0]))
    scaling = np.zeros(V.shape[0])
 
    # setting beta(T) = 1
    unscaled_beta[V.shape[0] - 1] = np.ones((a.shape[0]))
    scaling[0] = np.sum(unscaled_beta[V.shape[0] - 1])
    beta[V.shape[0] - 1] = unscaled_beta[V.shape[0] - 1] /scaling[0]
 
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            unscaled_beta[t, j] = (beta[t + 1] * norm.pdf(V[t + 1], 
                            loc=my_mean[j], scale=np.sqrt(my_cov[j]))).dot(a[j, :])
        scaling[t] = np.sum(unscaled_beta[t,:])
        beta[t, :] = unscaled_beta[t,:] / scaling[t]
 
    return beta
  
def Gauss_baumwelch(V, a, my_mean, my_cov, initial_distribution, n_iter=100, threshold = 0.001):
    
    #perform Baum-welch updating mean and variance instead of B
    
    M = a.shape[0]
    T = len(V)
    likelihood = []
 
    for n in range(n_iter):
        
        alpha, like = Gauss_forward(V, a, my_mean, my_cov, initial_distribution)
        likelihood.append(like)
        beta = Gauss_backward(V, a, my_mean, my_cov,)
 
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * norm.pdf(V[t + 1], 
                            loc=my_mean, scale=np.sqrt(my_cov)).T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * norm.pdf(V[t + 1], 
                            loc=my_mean[i], scale=np.sqrt(my_cov[i])).T * beta[t + 1, i].T
                xi[i, :, t] = numerator / denominator
 
        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
 
        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        
        initial_distribution = gamma[:,0] / np.sum(gamma[:,0])
        
        #K = a.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(M):
            my_mean[l] = np.sum(gamma[l,:] * V) / np.sum(gamma[l,:])  
            #update means
            my_cov[l] = np.sum(gamma[l,:-1] * (V[:-1] - my_mean[l])**2) / np.sum(gamma[l,:-1])
            #update variance
            
        #when threshold is hit, stop algorithm
        if len(likelihood) > 2 and abs(likelihood[-1] - likelihood[-2]) < threshold: 
            break
 
    return {"a":a, 
            "mu":my_mean, 
            "sigma":my_cov, 
            "pi":initial_distribution, 
            "likelihood":likelihood}


means = [30, 40]
covs = [5,5]
newV = np.array(myGC)

alpha = Gauss_forward(newV, new_param["a"], means, covs, new_param["mu"])[0]
beta = Gauss_backward(newV, new_param["a"], means, covs)

#perform Bum-welch
cont_new_params = Gauss_baumwelch(newV, new_param["a"], means, covs, new_param["mu"], n_iter=30)

#perform MAP on new parameters
nalpha = Gauss_forward(newV, cont_new_params["a"], cont_new_params["mu"], 
                       cont_new_params["sigma"], cont_new_params["sigma"])[0]
nbeta = Gauss_backward(newV, new_param["a"],
                       cont_new_params["mu"], cont_new_params["sigma"])

denominator = np.sum(nalpha*nbeta, 1)
nposterior = np.zeros((nbeta.shape))
for i in range(len(denominator)):
    nposterior[i,:] = nalpha[i,:] *nbeta [i,:]/denominator[i]
    
fig, axs = plt.subplot_mosaic("AB",figsize=(10,6))    #get mosaic plot 

axs["A"].plot(cont_new_params["likelihood"], color = '#4BC5D4')
axs["A"].set_title(f"Gaussian: Log-likelihood = {round(np.max(cont_new_params['likelihood']),2)}",
                   **hfont, size = 20)
axs["A"].set_ylabel("Iterations",**hfont,size = 15)
axs["A"].set_xlabel("Log-likelihood",**hfont,size = 15)
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
         size=20, weight='bold')    

axs["B"].scatter(range(len(myGC[:500])), myGC[:500], 
                 c=np.argmax(nposterior,1)[:500], cmap = 'tab20b', alpha = 0.7)
axs["B"].legend(handles = [lab1, lab2], title = "Hidden States", loc = 0)
axs["B"].set_title("Gaussian HMM for GC content",**hfont,size = 20)           
axs["B"].set_ylabel("Emission and hidden state spaces",**hfont,size = 15)
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
         size=20, weight='bold')    
fig.tight_layout(h_pad = 1)       #tight layout 

plt.savefig('Q6.png',bbox_inches='tight')   #save figure


                                ### END OF CODE ###


