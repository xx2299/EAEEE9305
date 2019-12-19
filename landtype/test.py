import os
import caffe
import numpy as np

deploy='deploy.prototxt'      
caffe_model='landtype_alexnet_train_iter_10000.caffemodel'
mean_file='mean.npy'
labels_filename='words.txt'

import os
dir='temp'
filelist=[]
filenames=os.listdir(dir)
filenames.sort(key=lambda x:int(x[:-4]))
for fn in filenames:
    fullfilename=os.path.join(dir,fn)
    filelist.append(fullfilename)
    

net=caffe.Net(deploy,caffe_model,caffe.TEST)
    
transformer=caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
transformer.set_mean('data',np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))  


for i in range(0,len(filelist)):
    img=filelist[i]       
    
    im=caffe.io.load_image(img) 
    net.blobs['data'].data[...]=transformer.preprocess('data',im) 
    
    out=net.forward()
    
    labels=np.loadtxt(labels_filename,str,delimiter='/t') 
    prob=net.blobs['prob'].data[0].flatten()   
    
    index1=prob.argsort()[-1]        
    
    print filenames[i],labels[index1],'--',prob[index1]   
