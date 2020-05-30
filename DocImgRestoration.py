import numpy as np 
import cv2 as cv
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from keras.models import Model, load_model
from keras.layers import Input, Dense
import math
from keras.layers import Conv2D, Conv2DTranspose,Dense, Dropout, Flatten, Input, MaxPooling2D, UpSampling2D
from keras.applications.vgg16 import VGG16
from keras_contrib.losses import DSSIMObjective 
import glob   
import time
import sys
#--------------------------------------main code-----------------------------------------------------------------------

model = load_model('model.h5',custom_objects={ 'DSSIMObjective': DSSIMObjective(kernel_size=23)})
model.load_weights('model_weights.h5')


def patchextractor(img,patchsize,stride):     
	patch=[];
	(a,b)=img.shape
	xlim=int(np.ceil(float(max(a,patchsize[0])-patchsize[0]+stride)/stride))*stride
	ylim=int(np.ceil(float(max(b,patchsize[1])-patchsize[1]+stride)/stride))*stride
	# print(xlim,ylim)

	for i in range(0,xlim,stride):    	
		for j in range(0,ylim,stride):	    	
			# print(i,j)
			ilim=min(i+patchsize[0],a)
			jlim=min(j+patchsize[1],b)

			temp=img[ i:ilim, j:jlim ]

			patch_tmp=np.ones(shape=patchsize,dtype='float32')
			patch_tmp[0:temp.shape[0],0:temp.shape[1]]=temp
			patch.append(patch_tmp)

	return patch

def reconstruct(patches,patchsize,imgsize,stride):
	a,b=imgsize
	xlim=int(np.ceil(float(max(a,patchsize[0])-patchsize[0]+stride)/stride))*stride
	ylim=int(np.ceil(float(max(b,patchsize[1])-patchsize[1]+stride)/stride))*stride
	c=0

	R=np.zeros((a,b)).astype("float32")#zeroes previously
	C=np.zeros((a,b)).astype("float32")#zeroes previously
	for i in range(0,xlim,stride):    	
		for j in range(0,ylim,stride):	    	
			# print('extract',i,j)
			ilim=min(i+patchsize[0],a)
			jlim=min(j+patchsize[1],b)

			# print(ilim,jlim,patches[c].shape)
			R[ i:ilim, j:jlim ]+=patches[c][ 0:ilim-i, 0:jlim-j]
			C[ i:ilim, j:jlim ]+=1.0

			c+=1
	# print(R)
	# print(C)
	R[C>0]=R[C>0]/C[C>0]
	# R=R.reshape(imgsize)
	return R


def extractText(im1):

	p=im1.shape
	patch_stride=10
	X1=patchextractor(im1,(256,256),patch_stride)
	X2=np.array(X1)
	X2=np.expand_dims(X1,-1)
	Y1=model.predict(X2)
	r=reconstruct(Y1[:,:,:,0],(256,256),p,patch_stride)

	#----find threshold----------------------

	dv={}
	img=np.array(r*255,dtype=int)
	vals=np.unique(img)
	for i in vals:
		dv[i]=0
	for i in (img.flatten()):
		dv[i]+=1
	sum=0
	tharr=[]
	arr=np.array(dv.keys())
	for i in range(2,arr.shape[0]-2):
		 tharr.append ((dv[arr[i]]+dv[arr[i-1]]+dv[arr[i-2]]+dv[arr[i+1]]+dv[arr[i+2]])/5) #--
	th=arr[np.argmin(np.array(tharr))+2]

	#----------------final image------------
	out_file_name="output.png"
	r1=(r*255)>th
	r1.dtype='uint8'
	
	# print('Text Extracted successfully')
	return r1



class GMM:
    
    def __init__(self, k = 4, eps = 0.0001):
        self.k = k ## number of clusters
        self.eps = eps ## threshold to stop `epsilon`
        
        # All parameters from fitting/learning are kept in a named tuple
        from collections import namedtuple
    
    def fit_EM(self, X, max_iters = 1000):
        
        # n = number of data-points, d = dimension of data points        
        n, d = X.shape
        
        # randomly choose the starting centroids/means 
        ## as 3 of the points from datasets        
        mu = X[np.random.choice(n, self.k, False), :]
        
        # initialize the covariance matrices for each gaussians
        Sigma= [np.eye(d)] * self.k
        
        # initialize the probabilities/weights for each gaussians
        w = [1./self.k] * self.k
        
        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for eack of k gaussians
        R = np.zeros((n, self.k))
        
        ### log_likelihoods
        log_likelihoods = []
        
        P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.)* np.exp(-.5 * np.einsum('ij, ij -> i',X - mu, np.dot(np.linalg.inv(s) , (X - mu).T).T ) ) 
                        
        # Iterate till max_iters iterations        
        while len(log_likelihoods) < max_iters:
            
            # E - Step
            
            ## Vectorized implementation of e-step equation to calculate the 
            ## membership for each of k -gaussians
            for k in range(self.k):
                R[:, k] = w[k] * P(mu[k], Sigma[k])

            ### Likelihood computation
            log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
            
            log_likelihoods.append(log_likelihood)
            
            ## Normalize so that the responsibility matrix is row stochastic
            R = (R.T / np.sum(R, axis = 1)).T
            
            ## The number of datapoints belonging to each gaussian            
            N_ks = np.sum(R, axis = 0)
            
            
            # M Step
            ## calculate the new mean and covariance for each gaussian by 
            ## utilizing the new responsibilities
            for k in range(self.k):
                
                ## means
                mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis = 1).T
                x_mu = np.matrix(X - mu[k])
                
                ## covariances
                Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
                
                ## and finally the probabilities
                w[k] = 1. / n * N_ks[k]
            # check for onvergence
            if len(log_likelihoods) < 2 : continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps: break
        
        ## bind all results together
        from collections import namedtuple
        self.params = namedtuple('params', ['mu', 'Sigma', 'w', 'log_likelihoods', 'num_iters'])
        self.params.mu = mu
        self.params.Sigma = Sigma
        self.params.w = w
        self.params.log_likelihoods = log_likelihoods
        self.params.num_iters = len(log_likelihoods)       
        
        return self.params
    
    def plot_log_likelihood(self):
        import pylab as plt
        plt.plot(self.params.log_likelihoods)
        plt.title('Log Likelihood vs iteration plot')
        plt.xlabel('Iterations')
        plt.ylabel('log likelihood')
        plt.show()
    
    def predict(self, x):
        p = lambda mu, s : np.linalg.det(s) ** - 0.5 * (2 * np.pi) ** (-len(x)/2) * np.exp( -0.5 * np.dot(x - mu ,np.dot(np.linalg.inv(s) , x - mu)))
        probs = np.array([w * p(mu, s) for mu, s, w in zip(self.params.mu, self.params.Sigma, self.params.w)])
        return probs/np.sum(probs)

input_file_name=sys.argv[1]
fname=input_file_name.split('.')[0]
# textout_file_name=fname+'_textout.png'
output_file_name=fname+'_output.png'
input_img=cv.imread(input_file_name,0)/255.0
input_img_color=cv.imread(input_file_name,1)/255.0

textout=extractText(input_img)
# cv.imwrite(textout_file_name,textout,[cv.IMWRITE_PNG_BILEVEL, 1])
textout.astype("float32")

input_img_color.astype("float32")
X=input_img_color
p=X.shape
Y=np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))

gmm = GMM()
params = gmm.fit_EM(Y,max_iters=100)

mu_sum=0.11*params.mu[:,0]+0.59*params.mu[:,1]+0.3*params.mu[:,2]

ep=-np.exp(-200)
chk=(input_img>0.9)
if(np.sum(chk)!=0):
	mu_sum[mu_sum.argmax()]=ep
bkgrnd_index=mu_sum.argmax()

Z=np.random.multivariate_normal(params.mu[bkgrnd_index],params.Sigma[bkgrnd_index],(p[0],p[1])).astype("float32")
Z=cv.GaussianBlur(Z,(7,7),cv.BORDER_DEFAULT)

#------------overlay on bakgrnd-----------------

chk=np.repeat(np.expand_dims(chk,axis=2),3,axis=2)
Z= Z*(1-chk) + chk*input_img_color

chk=textout<1
chk=np.repeat(np.expand_dims(chk,axis=2),3,axis=2)
Z= input_img_color*chk + (1-chk)*Z

Z=cv.cvtColor(Z.astype('float32'),cv.COLOR_BGR2RGB)

plt.imsave(output_file_name,Z)
print('Restoration done successfully')




