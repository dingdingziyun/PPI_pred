import numpy as np
from sklearn import svm


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


out_performance=open('../output/scores','w')
out_performance.write('recall'+'\t'+'precision'+'\t'+'specificity'+'\t'+'F1 score'+'\n')
#print 'recall'+'\t'+'precision'+'\t'+'specificity'+'\t'+'F1 score'

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
    #print out_ts_data_neg.shape
    #svc=svm.SVC(C=10.0, kernel='rbf', gamma=1.0)
    print "finished splitting the outer loop data "+str(fold+1)
    svc=svm.SVC(C=0.001, kernel='poly', gamma=100)
    svc.fit(out_tr_data, out_tr_label)
    #print float(sum(svc.predict(out_ts_data)==out_ts_label))/len(out_ts_label)

    tp=sum(svc.predict(out_ts_data_pos)==out_ts_label_pos)
    tn=sum(svc.predict(out_ts_data_neg)==out_ts_label_neg)
    fp=len(out_ts_data_neg)-tn
    fn=len(out_ts_label_pos)-tp
    #print tp
    #print tp*1.0/(tp+fn)

    recall=tp*1.0/(tp+fn)
    #print recall
    precision=tp*1.0/(tp+fp)
    #print precision
    specificity=tn*1.0/(tn+fp)
    #print specificity
    F1=2*tp*1.0/(2*tp+fp+fn)
    #print F1
    #print str(recall)+'\t'+str(precision)+'\t'+str(specificity)+'\t'+str(F1)
    print "finished compute the scores"
    out_performance.write(str(recall)+'\t'+str(precision)+'\t'+str(specificity)+'\t'+str(F1)+'\n')
out_performance.close()
print "Done"
