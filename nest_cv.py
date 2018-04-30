from sklearn import svm
import numpy as np


# Load the dataset
my_pos_data= np.genfromtxt('../dataset/ara_svm_input',delimiter=',')
my_neg_data= np.genfromtxt('../dataset/neg_svm_input',delimiter=',') 

X_pos = my_pos_data[:, 1:]
y_pos = my_pos_data[:, 0]
X_neg = my_neg_data[:, 1:]
y_neg = my_neg_data[:, 0]
'''
#comment out to test the smaller dataset
X_pos = my_pos_data[0:100, 1:]
y_pos = my_pos_data[0:100, 0]
X_neg = my_neg_data[0:100, 1:]
y_neg = my_neg_data[0:100, 0]
'''

X = np.concatenate((X_pos, X_neg), axis=0)
y = np.concatenate((y_pos, y_neg), axis=0)
print "finished loading data"

c_range=list(np.logspace(-3, 2, num=6, base=10, endpoint=True))
gamma_range=list(np.logspace(-3, 2, num=6, base=10, endpoint=True))
kernel=['linear', 'rbf', 'poly']


out_performance=open('../output/performance','w')
best_param=['rbf', 1, 1]
best_acc=0

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

    print "finished splitting the outer loop data: outer fold "+str(fold+1)
    output=open('../output/fold_'+str(fold+1)+'_acc','w')
    for k in kernel:
        for c in c_range:
            for g in gamma_range:
                acc=0
                svc = svm.SVC(C=c, kernel=k, gamma=g)
                for infold in range(0, 4):
                    in_ts_data_pos=out_ts_data_pos[infold*(len(out_ts_data_pos)/4):((infold+1)*(len(out_ts_data_pos)/4)),:]
                    in_ts_data_neg=out_ts_data_neg[infold*(len(out_ts_data_neg)/4):((infold+1)*(len(out_ts_data_neg)/4)),:]
                    in_tr_data_pos=np.delete(out_tr_data_pos, np.s_[infold*(len(out_tr_data_pos)/4):((infold+1)*(len(out_tr_data_pos)/4))], axis=0)
                    in_tr_data_neg=np.delete(out_tr_data_neg, np.s_[infold*(len(out_tr_data_neg)/4):((infold+1)*(len(out_tr_data_neg)/4))], axis=0)
                    
                    in_ts_label_pos=out_ts_label_pos[infold*(len(out_ts_label_pos)/4):((infold+1)*(len(out_ts_label_pos)/4))]
                    in_ts_label_neg=out_ts_label_neg[infold*(len(out_ts_label_neg)/4):((infold+1)*(len(out_ts_label_neg)/4))]
                    in_tr_label_pos=np.delete(out_tr_label_pos, np.s_[infold*(len(out_tr_label_pos)/4):((infold+1)*(len(out_tr_label_pos)/4))], axis=0)
                    in_tr_label_neg=np.delete(out_tr_label_neg, np.s_[infold*(len(out_tr_label_neg)/4):((infold+1)*(len(out_tr_label_neg)/4))], axis=0)

                    in_ts_data=np.concatenate((in_ts_data_pos, in_ts_data_neg), axis=0)
                    in_tr_data=np.concatenate((in_tr_data_pos, in_tr_data_neg), axis=0)
                    in_ts_label=np.concatenate((in_ts_label_pos, in_ts_label_neg), axis=0)
                    in_tr_label=np.concatenate((in_tr_label_pos, in_tr_label_neg), axis=0)
                    print "finished splitting the inner loop data: inner fold "+str(infold+1)
                    svc.fit(in_tr_data, in_tr_label)
                    acc=acc+float(sum(svc.predict(in_ts_data)==in_ts_label))/len(in_ts_label)
                #print str(fold+1)+'\t'+k+'\t'+str(c)+'\t'+str(g)+'\t'+str(acc/4)
                output.write(str(fold+1)+'\t'+k+'\t'+str(c)+'\t'+str(g)+'\t'+str(acc/4)+'\n')

                if (acc/4) > best_acc:
                    best_acc=acc/4
                    best_param=[k, c, g]

    output.write('best_parameters'+'\t'+str(best_param[0])+'\t'+str(best_param[1])+'\t'+str(best_param[2]))
    output.close()
    print 'the best kernel is ' +best_param[0]+', the best C is'+str(best_param[1])+', the best gamma is '+str(best_param[2])

    #svc=svm.SVC(C=0.001, kernel='poly', gamma=100)
    svc=svm.SVC(C=best_param[1], kernel=best_param[0], gamma=best_param[2])
    svc.fit(out_tr_data, out_tr_label)
    
    
    
    #print float(sum(svc.predict(out_ts_data)==out_ts_label))/len(out_ts_label)
    out_performance.write(str(float(sum(svc.predict(out_ts_data)==out_ts_label))/len(out_ts_label))+'\n')
out_performance.close()
print "Done"