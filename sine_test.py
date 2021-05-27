import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def create_sine(fSampling, freq1, freq2):
    tSample = 1 / fSampling
    T = np.arange(0, 1 - tSample, tSample)  # 1s
    #A1 = 3
    A1 = np.random.uniform(2, 4)   
    NoiseAmp = 0
    noise = NoiseAmp * (np.random.rand(T.size) - 0.5) 
    if freq2 == 0:        
        Sig = A1 * np.sin(2 * np.pi * freq1 * T) + noise
    else:
        Sig = (A1 * np.sin(2 * np.pi * freq1 * T) + A1 * np.sin(2 * np.pi * freq2 * T))/2 + noise
    return Sig


if __name__ == '__main__':
    fSampling = 20000 # 20 kHz - length of each signal
    freq1 = 9999  # 10kHz - freq of first sine
    freq2 = 12000  # 12kHz - freq of second sine
    numOfSamples = 5000
    one_sine_data = np.zeros([numOfSamples, fSampling-1], dtype=np.float)
    two_sine_data = np.zeros([numOfSamples, fSampling-1], dtype=np.float)
    for i in range(numOfSamples):
        # create 5000 sines of freq 10kHz with noise
        one_sine_data[i] = create_sine(fSampling, freq1, 0)  
        # create 5000 sines of freq 10kHz + 12kHz with noise
        two_sine_data[i] = create_sine(fSampling, freq1, freq2)

    tSample = 1 / fSampling
    T = np.arange(0, 1 - tSample, tSample)  # 1s
    plt.figure()
    plt.plot(T, one_sine_data[1])
    #plt.figure()
    #plt.plot(T, one_sine_data[4999])
    plt.show()
    
    # get FFT for signals
    mid = int(fSampling/2)
    one_sine_fft = np.zeros([numOfSamples, mid-1])
    two_sine_fft= np.zeros([numOfSamples, mid-1])
    # Get half of |FFT|^2 
    for i in range(numOfSamples): 
        one_sine_fft[i] = np.power(np.absolute(sp.fft.fft(one_sine_data[i])),2)[mid:]
        two_sine_fft[i] = np.power(np.absolute(sp.fft.fft(two_sine_data[i])),2)[mid:]

    f = np.arange(0, mid-1, 1)
    plt.figure()
    plt.plot(f, two_sine_fft[0])
    plt.show() 
    
    
    #%% Arrange Data
    
    # Add each signal a bit to identify whether it's one or two
    one_sine_fft_bit = np.hstack((one_sine_fft,np.ones([numOfSamples,1])))
    two_sine_fft_bit = np.hstack((two_sine_fft,np.ones([numOfSamples,1])+1))
    n_samples = numOfSamples  
    # Build DB 90% one_sine, 10% two_sine
    mainSlice = int(numOfSamples*0.9)
    secSlice = int(numOfSamples*0.1)
    combined_fft = np.vstack((one_sine_fft_bit[0:mainSlice,:],two_sine_fft_bit[0:secSlice,:]))
    
    # Shuffle DB
    rand_gen = np.random.RandomState(0)
    indices = np.arange(n_samples)
    rand_gen.shuffle(indices)   
    combined_fft_shuffled = combined_fft[indices]
    
    #%%  One Class SVM Classification
    
    from sklearn import svm
    
    # Build SVM, fit, predict
    clf = svm.OneClassSVM(nu=0.1, kernel = 'linear', max_iter = 1000000)
    clf.fit(combined_fft_shuffled[:,0:9999])
    labels = clf.predict(combined_fft_shuffled[:,0:9999])
    anom_index = np.transpose(np.asarray(np.where(labels == -1)))
        
    
    #%% K-Means Classification
    
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(combined_fft_shuffled[:,0:9999])
    labels = kmeans.labels_
    
    # Make the bigger cluster have a label of 1, and the smaller cluster -1
    sizeLabels = np.size(labels)
    sizeLabels1 = np.count_nonzero(labels==1)
    if (sizeLabels1 > sizeLabels/2):
        labels[labels == 0] = -1
    else:
        labels[labels == 1] = -1
        labels[labels == 0] = 1
    
    anom_index = np.transpose(np.asarray(np.where(labels == -1)))
    
    #%% OCSVM - Testing nu
    
    from sklearn import svm    
    
    nus = np.arange(0.05,0.95,0.05)
    numOfAnom = np.zeros(np.size(nus))
    # Build SVM, fit, predict for different nu values
    for i,curr_nu in enumerate(nus):
        clf = svm.OneClassSVM(nu = curr_nu, kernel = 'linear')
        labels = clf.fit_predict(one_sine_fft)
        anom_index = np.transpose(np.asarray(np.where(labels == -1)))
        numOfAnom[i] = np.size(anom_index)
    
    
    #%% Calculate FP, FN
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    numAnom = len(anom_index)
    for i in range(n_samples):
        if(combined_fft_shuffled[i,9999] == 1 and labels[i] == -1):
            cnt1 += 1
        if(combined_fft_shuffled[i,9999] == 2 and labels[i] == 1):
            cnt2 += 1            
    print(cnt1, cnt2)
    
    FN = cnt1/numAnom
    FP = cnt2/numAnom
    print('FN = ',FN,', FP = ', FP)
    
    
    
    
    
    