import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import svm


def create_sine(fSampling, freq, withNoise):
    tSample = 1 / fSampling
    T = np.arange(0, 1 - tSample, tSample)  # 1s
    Offset = 0  # 1.5
    A1 = 3
    NoiseAmp = 1e6
    noise = NoiseAmp * (np.random.rand(T.size) - 0.5) if withNoise else 0
    Sig = Offset + A1 * np.sin(2 * np.pi * freq * T) + noise
    # plt.plot(T, A1/2 * np.sin(2*np.pi*12000*T) + A1/2 * np.sin(2*np.pi*9999*T))
    # plt.show()
    return Sig


if __name__ == '__main__':
    fSampling = 20000  # 20 kHz - length of each signal
    freq1 = 9999  # 10kHz - freq of first sine
    freq2 = 12000  # 12kHz - freq of second sine
    one_sine_data = np.zeros([5000, fSampling - 1], dtype=np.float)
    two_sine_data = np.zeros([5000, fSampling - 1], dtype=np.float)
    for i in range(5000):
        # create 5000 sines of freq 10kHz with noise
        one_sine_data[i] = create_sine(fSampling, freq1, 1)  
        # create 5000 sines of freq 10kHz + 12kHz with noise
        two_sine_data[i] = (create_sine(fSampling, freq1, 1) + create_sine(fSampling, freq2, 0)) / 2  

    tSample = 1 / fSampling
    T = np.arange(0, 1 - tSample, tSample)  # 1s
    plt.figure()
    plt.plot(T, one_sine_data[0])
    #plt.figure()
    #plt.plot(T, one_sine_data[4999])
    plt.show()
    
    # get FFT for signals
    mid = int(fSampling/2)
    one_sine_fft = np.zeros([5000, mid - 1])
    two_sine_fft= np.zeros([5000, mid - 1])
    # Get half of |FFT|^2 
    for i in range(5000): 
        one_sine_fft[i] = np.power(np.absolute(sp.fft.fft(one_sine_data[i])),2)[mid:]
        two_sine_fft[i] = np.power(np.absolute(sp.fft.fft(two_sine_data[i])),2)[mid:]

    f = np.arange(0, mid - 1, 1)
    plt.figure()
    plt.plot(f, two_sine_fft[100])
    plt.show()

    # One Class SVM Classification
    

    
    #%%
    # Add each signal a bit to identify whether it's one or two
    one_sine_fft_bit = np.hstack((one_sine_fft,np.ones([5000,1])))
    two_sine_fft_bit = np.hstack((two_sine_fft,np.ones([5000,1])+1))
    n_samples = 5000  
    # Build DB 90% one_sine, 10% two_sine
    combined_fft = np.vstack((one_sine_fft_bit[0:4500,:],two_sine_fft_bit[0:500,:]))
    
    # Shuffle DB
    rand_gen = np.random.RandomState(0)
    indices = np.arange(n_samples)
    rand_gen.shuffle(indices)   
    combined_fft_shuffled = combined_fft[indices]
    
    # Build SVM, fit, predict
    clf = svm.OneClassSVM(nu=0.1, kernel = 'linear')
    clf.fit(combined_fft_shuffled[:,0:9999])
    pred = clf.predict(combined_fft_shuffled[:,0:9999])
    anom_index = np.transpose(np.asarray(np.where(pred == -1)))
    
    #%% Calculate FP, FN
    cnt1 = 0
    cnt2 = 0
    numAnom = len(anom_index)
    for i in range(n_samples):
        if(combined_fft_shuffled[i,9999] == 1 and pred[i] == -1):
            cnt1 += 1
        if(combined_fft_shuffled[i,9999] == 2 and pred[i] == 1):
            cnt2 += 1
    print(cnt1, cnt2)
    
    FN = cnt1/numAnom
    FP = cnt2/numAnom
    print('FN = ',FN,', FP = ', FP)
    
# =============================================================================
#     #%% 
#     n_samples = 5000  
#     # Build DB 90% one_sine, 10% two_sine
#     combined_fft = np.vstack((one_sine_fft[0:4500,:],two_sine_fft[0:500,:]))
#     
#     
#     # Build SVM, fit, predict
#     clf = svm.OneClassSVM(nu=0.1, kernel = 'linear')
#     clf.fit(combined_fft)
#     pred = clf.predict(combined_fft)
#     anom_index = np.transpose(np.asarray(np.where(pred == -1)))
#     
#     cnt1 = 0
#     for i in range(4500):
#         if(pred[i] != 1):
#             cnt1 += 1
#     
#     cnt2 = 0
#     for i in range(4500,5000):
#         if(pred[i] != -1):
#             cnt2 += 1
#     
#     numAnom = len(anom_index)
#     FN = cnt1/numAnom
#     FP = cnt2/numAnom  
#     
#     print('FN = ',FN,', FP = ', FP)
# =============================================================================
    
    #%%
    # =============================================================================
#     rand_gen = np.random.RandomState(0)
#     indices = np.arange(n_samples)
#     rand_gen.shuffle(indices)
#     
#     combined_fft_shuffled = combined_fft[indices]
# =============================================================================
    #one_sine_fft_t = np.transpose(one_sine_fft)
# =============================================================================
#     clf1 = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
#     clf1.fit(one_sine_fft)
#     pred1 = clf1.predict(one_sine_fft)
# 
#     anom_index_one = np.where(pred1 == -1)
#     values_one = one_sine_fft[anom_index_one]    
#     
#     scores_one = clf1.score_samples(one_sine_fft)
#     
#     #two_sine_fft_t = np.transpose(two_sine_fft)
#     clf2 = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
#     clf2.fit(two_sine_fft)
#     pred2 = clf2.predict(two_sine_fft)
# 
#     anom_index_two = np.where(pred2 == -1)
#     values_two = two_sine_fft[anom_index_two]    
#     
#     scores_two = clf2.score_samples(two_sine_fft)
# =============================================================================
    
# =============================================================================
#     np.savetxt('anom_index.txt', np.transpose(anom_index))
#     np.savetxt('values.txt', values)
# =============================================================================

# =============================================================================
#     plt.scatter(one_sine_fft[0,:], one_sine_fft[1,:])
#     plt.scatter(values[0,:], values[1,:], color='r')
#     plt.show()
# =============================================================================
