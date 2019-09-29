# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 00:03:15 2019

@author: aleal
"""
#from sklearn import datasets  
#import pandas as pd
#iris = datasets.load_iris()
#x = pd.DataFrame(iris['data'],columns=iris['feature_names'])
#print(x)
import pickle
import numpy as np
np.set_printoptions(threshold=np.inf)
#with open("C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\games\\arkanoid\\log\\2019-09-26_00-42-06.pickle", 'rb") as f1:
 #   data_list1 = pickle.load(f1)
with open("C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\games\\arkanoid\\log\\第三關-改球\\2019-09-27_21-31-55.pickle", "rb") as f1:
    data_list1 = pickle.load(f1)
    #print(data_list1[0])
with open("C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\games\\arkanoid\\log\\第三關-改球\\2019-09-27_21-34-00.pickle", "rb") as f2:
    data_list2 = pickle.load(f2)
   # print(data_list2[0])
with open("C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\games\\arkanoid\\log\\第三關-改球\\2019-09-27_21-34-45.pickle", "rb") as f3:
    data_list3 = pickle.load(f3)
  #  print(data_list3[0])
with open("C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\games\\arkanoid\\log\\第三關-改球\\2019-09-27_21-34-24.pickle", "rb") as f4:
    data_list4 = pickle.load(f4)  
    
with open("C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\games\\arkanoid\\log\\第二關-改球\\2019-09-27_22-02-56.pickle", "rb") as f5:
    data_list5 = pickle.load(f5)
    #print(data_list1[0])
with open("C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\games\\arkanoid\\log\\第二關-改球\\2019-09-27_22-03-20.pickle", "rb") as f6:
    data_list6 = pickle.load(f6)
   # print(data_list2[0])
with open("C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\games\\arkanoid\\log\\第二關-改球\\2019-09-27_22-03-37.pickle", "rb") as f7:
    data_list7 = pickle.load(f7)
  #  print(data_list3[0])
with open("C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\games\\arkanoid\\log\\第二關-改球\\2019-09-27_22-03-52.pickle", "rb") as f8:
    data_list8 = pickle.load(f8) 

Frame=[]
Status=[]
Ballposition=[]
PlatformPosition=[]
Bricks=[]
for ai in range(0 , 3):
    for i in range(0 , len(data_list1)):
        Frame.append(data_list1[i].frame)
        Status.append(data_list1[i].status)
        Ballposition.append(data_list1[i].ball)
        PlatformPosition.append(data_list1[i].platform)
        
    for i in range(0 , len(data_list2)):
        Frame.append(data_list2[i].frame)
        Status.append(data_list2[i].status)
        Ballposition.append(data_list2[i].ball)
        PlatformPosition.append(data_list2[i].platform)
        
    for i in range(0 , len(data_list3)):
        Frame.append(data_list3[i].frame)
        Status.append(data_list3[i].status)
        Ballposition.append(data_list3[i].ball)
        PlatformPosition.append(data_list3[i].platform)
    for i in range(0 , len(data_list4)):
        Frame.append(data_list4[i].frame)
        Status.append(data_list4[i].status)
        Ballposition.append(data_list4[i].ball)
        PlatformPosition.append(data_list4[i].platform)
        
    for i in range(0 , len(data_list5)):
        Frame.append(data_list5[i].frame)
        Status.append(data_list5[i].status)
        Ballposition.append(data_list5[i].ball)
        PlatformPosition.append(data_list5[i].platform)
        
    for i in range(0 , len(data_list6)):
        Frame.append(data_list6[i].frame)
        Status.append(data_list6[i].status)
        Ballposition.append(data_list6[i].ball)
        PlatformPosition.append(data_list6[i].platform)
        
    for i in range(0 , len(data_list7)):
        Frame.append(data_list7[i].frame)
        Status.append(data_list7[i].status)
        Ballposition.append(data_list7[i].ball)
        PlatformPosition.append(data_list7[i].platform)
        
    for i in range(0 , len(data_list8)):
        Frame.append(data_list8[i].frame)
        Status.append(data_list8[i].status)
        Ballposition.append(data_list8[i].ball)
        PlatformPosition.append(data_list8[i].platform)
    

import numpy as np
#/5 每次移動5 [:0]只取x
PlatX=np.array(PlatformPosition)[:,0][:,np.newaxis]
#print(PlatX)
PlatX_next=PlatX[1:,:]
#print(PlatX_next)
PlatY=np.array(PlatformPosition)[:,0][:,np.newaxis]

instruct=(PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5
#print(instruct)

BallX=np.array(Ballposition)[:,0][:,np.newaxis]
#print(PlatX)
BallX_next=BallX[1:,:]
print(len(BallX))
VX=(BallX_next-BallX[0:len(BallX_next),0][:,np.newaxis])

BallY=np.array(Ballposition)[:,1][:,np.newaxis]
#print(PlatX)
BallY_next=BallY[1:,:]
print(len(BallX))
VY=(BallY_next-BallY[0:len(BallY_next),0][:,np.newaxis])

#不取最後一個
Ballarray=np.array(Ballposition)[:-1]
print(len(Ballarray))
#特徵X
#x=np.hstack((Ballarray, PlatX[0:-1,0][:,np.newaxis],PlatY[0:-1,0][:,np.newaxis],VX,VY))
#特徵球x、y 平板x 球vx vy
x=np.hstack((Ballarray, PlatX[0:-1,0][:,np.newaxis],VX,VY))
print(len(PlatX[0:-1,0][:,np.newaxis]))
print(x)
y=instruct


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=999)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

yknn_bef_scaler=knn.predict(x_test)
acc_knn_bef_scaler=accuracy_score(yknn_bef_scaler,y_test)
print(acc_knn_bef_scaler)
#print(acc_knn_bef_scaler)

#from sklearn.preprocessing import StandardScaler
#scaler=StandardScaler()
#scaler.fit(x_train)
#x_train_stdnorm=scaler.transform(x_train)
#knn.fit(x_train_stdnorm,y_train)
#x_test_standorm=scaler.transform(x_test)
#yknn_aft_scaler=svm.predict(x_test_standorm)
#acc_knn_aft_scaler=accuracy_score(yknn_aft_scaler, y_test)

filename ="C:\\Users\\aleal\\OneDrive\\桌面\\MLGame-master\\MLGame-master\\knn_ex.sav"
pickle.dump(knn,open(filename, 'wb'))










