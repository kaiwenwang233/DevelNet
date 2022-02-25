
from detect_peak import detect_peaks
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
files = [f for f in glob.glob('./test_pred/results/' + "*.npz", recursive=True)]
file = open('./test_pred/picksfile.txt','a') 
for ifile in range(len(files)):
    preds=np.load('./test_pred/results/'+files[ifile].split('/')[-1][0:-4]+'.npz')
    ind=detect_peaks(preds['pred'][0,:,1],mph=0.5,mpd=10)
    pslabel=sio.loadmat('./STEADtraining/'+files[ifile].split('/')[-1][0:-4])['pslabel']
    labels=np.zeros(387)
    if pslabel.shape[1]==4:
        labels[np.min(pslabel[:,0]):np.max(pslabel[:,1])]=1
        labels[np.min(pslabel[:,2]):np.max(pslabel[:,3])]=1
    elif pslabel.shape[1]==2:
        labels[np.min(pslabel[:,0]):np.max(pslabel[:,1])]=1     
    indtrue=detect_peaks(labels,mph=0.5,mpd=10)
    file.write(files[ifile].split('/')[-1][0:-4]+'\n')
    file.write(str(indtrue).replace('\n', '')[1:-1]+'\n')
    file.write(str(ind).replace('\n', '')[1:-1]+'\n')
file.close() 
