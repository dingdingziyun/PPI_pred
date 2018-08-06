import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

def my_roc_curve(y_true, y_score, pos_label=1, sample_weight=None):  
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = (y_true == pos_label)
    y_true = y_true[desc_score_indices]
    weight = 1.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true * weight)[threshold_idxs]
    
    fps = 1 + threshold_idxs - tps
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, y_score[threshold_idxs], auc



# Load the dataset
my_pos_data= np.genfromtxt('../dataset/ara_svm_input',delimiter=',')
my_neg_data= np.genfromtxt('../dataset/neg_svm_input',delimiter=',') 
X_pos = my_pos_data[:, 1:]
y_pos = my_pos_data[:, 0]
X_neg = my_neg_data[:, 1:]
y_neg = my_neg_data[:, 0]



X = np.concatenate((X_pos, X_neg), axis=0)
y = np.concatenate((y_pos, y_neg), axis=0)
print "finished loading data"
out_auc=open('../output/auc', 'w')
for fold in range(0, 4):
    out_ts_data=np.nan
    out_tr_data=np.nan
    out_ts_label=np.nan
    out_tr_label=np.nan
   

    out_ts_data_pos=X_pos[fold*(len(X_pos)/4):((fold+1)*(len(X_pos)/4)),:]
    out_ts_data_neg=X_neg[fold*(len(X_neg)/4):((fold+1)*(len(X_neg)/4)),:]
    out_tr_data_pos=np.delete(X_pos, np.s_[fold*(len(X_pos)/4):((fold+1)*(len(X_pos)/4))], axis=0)
    out_tr_data_neg=np.delete(X_neg, np.s_[fold*(len(X_neg)/4):((fold+1)*(len(X_neg)/4))], axis=0)
    out_ts_label_pos=y_pos[fold*(len(y_pos)/4):((fold+1)*(len(y_pos)/4))]
    out_ts_label_neg=y_neg[fold*(len(y_neg)/4):((fold+1)*(len(y_neg)/4))]
    out_tr_label_pos=np.delete(y_pos, np.s_[fold*(len(y_pos)/4):((fold+1)*(len(y_pos)/4))], axis=0)
    out_tr_label_neg=np.delete(y_neg, np.s_[fold*(len(y_neg)/4):((fold+1)*(len(y_neg)/4))], axis=0)
    
    out_ts_data=np.concatenate((out_ts_data_pos, out_ts_data_neg), axis=0)
    out_tr_data=np.concatenate((out_tr_data_pos,out_tr_data_neg), axis=0)
   
    out_ts_label=np.concatenate((out_ts_label_pos, out_ts_label_neg), axis=0)
    out_tr_label=np.concatenate((out_tr_label_pos, out_tr_label_neg), axis=0)
    print "finished splitting the outer loop data "+str(fold+1)
    #svc=svm.SVC(C=10.0, kernel='rbf', gamma=1.0, probability=True)
    svc=svm.SVC(C=0.001, kernel='poly', gamma=100, probability=True)
    svc.fit(out_tr_data, out_tr_label)
   
    val=svc.predict_proba(out_ts_data)[:,1]
    a,b,t, auc = my_roc_curve(out_ts_label,val)
    out_auc.write('fold_'+str(fold+1)+'\t'+str(auc)+'\n')
    #print auc
    print "finished calculate scores for roc and AUC "+str(fold+1)
    #np.savetxt("../output/fold_a"+str(fold+1), a)
    #np.savetxt("../output/fold_b"+str(fold+1), b)
    
    plt.clf()
    plt.plot(a, b, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver operating characteristic outer fold '+str(fold+1))
    plt.ylabel('False positive rate')
    plt.xlabel('True positive rate')
    plt.savefig('../output/ROC_'+str(fold+1))
    print "finished plot fold "+str(fold+1)

print "Done"
