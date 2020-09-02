import matplotlib.pyplot as plt
import matplotlib.colors as mcl
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial import distance
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_curve, auc
import random

def detectMD(splited_train, test_data, splited_label, k, out_label):
    
    since = time.time()

    out_predict = []
    distance_list = []

    threshold = 1.0 
        
    cnt = 1

    print(len(test_data))

    for index, test_datum in enumerate(test_data):
        min_class_dist = 99999

        is_out = True

        print(index+1)
        

        for sindex, splited_datum in enumerate(splited_train):
            #print("aaaa: {}".format(sindex))

            maxdist = getMaxDistance(splited_datum, test_datum)

            
            #print(sindex)
            if maxdist < threshold:
                is_out = False
                break
        
        out_predict.append(is_out)             
    
    time_elapsed = time.time() - since

    print(time_elapsed)  

    
    tn, fp, fn, tp = confusion_matrix(out_label, out_predict).ravel()
  

    print('Total: {0}'.format(test_data.shape[0]))
    print('TN: {0}'.format(tn))
    print('FP: {0}'.format(fp))
    print('FN: {0}'.format(fn))
    print('TP: {0}'.format(tp))    
    
    
def detectAV(splited_train, test_data, splited_label, k, out_label):
    
    since = time.time()

    cnt = 1

    out_predict = []
    dist_list = []

    # threshold
    threshold = 0.93278782
    
    mcd_list = []

    # test data loop
    for index, test_datum in enumerate(test_data):
        min_class_dist = 99999

        is_out = True

        print(index+1)

        # reference data loop for each class
        for sindex, splited_datum in enumerate(splited_train):
            
            # get average distance for one class
            avgdist = getAvDistance(splited_datum, test_datum, k)          

            # get minimum class distance
            if avgdist < min_class_dist:
                min_class_dist = avgdist
        
        #check whether the sample is a outlier
        if min_class_dist < threshold:
            is_out = False
            #break
        

        dist_list.append(min_class_dist)
        out_predict.append(is_out)       
    
        
    time_elapsed = time.time() - since

    print(time_elapsed)
    
    # True Negatives, True Positives, False Negatives, True Positives
    tn, fp, fn, tp = confusion_matrix(out_label, out_predict).ravel()
  

    print('Total: {0}'.format(test_data.shape[0]))
    print('TN: {0}'.format(tn))
    print('FP: {0}'.format(fp))
    print('FN: {0}'.format(fn))
    print('TP: {0}'.format(tp))

    # get roc_auc score
    auc_score = roc_auc_score(out_label, dist_list)
    print('auc: {}'.format(auc_score))
   
    # True Postive rate False Positive rate, threshold list
    fpr, tpr, th_list = roc_curve(out_label, dist_list)
    
    # get auc area
    roc_auc = auc(fpr, tpr)

    '''
    print(fpr)
    print(tpr)
    print(th_list)
    '''


    # draw roc cruve

    lw = 2

    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    

    

def detectCent(splited_train, test_data, splited_label, k, out_label):
 

    cnt = 1        
        
    out_predict = []

    threshold = 1.0  

    mcd_list = []

    for test_datum in test_data:
        min_class_dist = 99999

        print(cnt)
        for sindex, splited_datum in enumerate(splited_train):                             

            centdist = getCentDistance(splited_datum, test_datum, k)

            #print(sindex)
            if min_class_dist > centdist:
                min_class_dist = centdist

        cnt += 1
        
        mcd_list.append(min_class_dist) 
   
    
    for mcd in mcd_list:
        if mcd > threshold:
            out_predict.append(True)
        else:
            out_predict.append(False)          

    
    tn, fp, fn, tp = confusion_matrix(out_label, out_predict).ravel()
     

    print('Total: {0}'.format(test_data.shape[0]))
    print('TN: {0}'.format(tn))
    print('FP: {0}'.format(fp))
    print('FN: {0}'.format(fn))
    print('TP: {0}'.format(tp))




def getCentDistance(data, point, k):

    summ = np.zeros(512)

    #print(len(data))
    
    for datum in data:        
        
        #print(datum)
        summ = summ + datum

    #print(summ)    
    cent = summ / k        
        
    centdist = np.linalg.norm(cent - point)

    return centdist

def getMaxDistance(splited_data, test_datum):

    maximum = -1
    #print(len(data))
    
    for splited_datum in splited_data:        

        dist = np.linalg.norm(splited_datum - test_datum)

        if dist > maximum:
            maximum = dist


    return dist

def getAvDistance(splited_data, test_datum, k):
    
    #print(len(data))

    sum_dist = 0    
    
    for splited_datum in splited_data:        
        
        #get L2 distance for each reference sample 
        dist = np.linalg.norm(splited_datum - test_datum)

        sum_dist += dist

    avg_dist = sum_dist / k

    #print(avg_dist)

    return avg_dist    

  

if __name__ == '__main__':

    #read csv files
    train_data = pd.read_csv('./baseline_500_ref.csv')
    test_data = pd.read_csv('./baseline_500_test.csv')

    class_num = 500

    # label for outliers (True or False)
    out_label = (test_data['label'] == (class_num + 1))

    splited_train = []
    splited_label = []

    # data list and label list for each class of reference data
    for label_index in range(1, (class_num + 1)):
        temp_frame = train_data.loc[train_data['label'] == label_index].drop(train_data.columns[0], axis=1).to_numpy()
        splited_train.append(temp_frame)
        splited_label.append(np.arange((temp_frame.shape[0])))
       
    # drop the label column for the test dataset
    test_data = test_data.drop(test_data.columns[0], axis=1)
    test_data = test_data.to_numpy()

    # detection of outliers    
    #detectMD(splited_train, test_data, splited_label, 5, out_label)
    detectAV(splited_train, test_data, splited_label,5, out_label)
    #detectCent(splited_train, test_data, splited_label, 5, out_label)
    
  
   

