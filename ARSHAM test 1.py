#!/usr/bin/env python
# coding: utf-8

# In[45]:


#importing all important functions
import pandas as pd
import matplotlib as mp
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
#reading excel file into python
df=pd.read_csv (r'C:\Users\xtran\OneDrive\Desktop\point-cloud.csv\point-cloud.csv',header=None)
#turning excel spreadsheet into python array
dff = df.values


# In[2]:


#plot setup
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

#x and y arrays for plotting 0-5000, 0-800
x_array = np.arange(0, 5000)
y_array = np.arange(0,800)

#setting blank arrays
xx=[]
yy=[]
zz=[]

#creating 1d array for 3dplot trisurf function
for x in x_array:
    for y in y_array:
        #appends x,y and z from datafile with corresponding indicies for evaluation in plotting
        xx.append(x)
        yy.append(y)
        zz.append(dff[x][y])
#plots the dataframe in 3D
ax.plot_trisurf(xx, yy, zz,cmap='Spectral', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()


# In[36]:


#jj blank array // searching for indicies to cut the sample using cutting threshold of 0.004 and appends index 800 at the end

jj=[]
for j in range(0,799):    
        if dff[2000,j]- dff[2000,j+1]>0.004:
            jj.append(j+1)
jj.append(800)


# In[185]:


#sets jj as accepted in next function array
jjj = np.asarray(jj)
jjj


# In[40]:


#g is an indicy for cutting the 3d plot, starting from zero
g=0

#jjj is an array of indicies for cutting that did not meet threshold of 0.004 for a difference test between two points
for f in range(0,len(jjj)-1):
    #because there are slopes and not just jumps, a threshold of 30 point difference is set for the plotting algorithm // (see jjj)
    if (jjj[f]-jjj[f+1])<-30:
        #plot setup
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        #sets x and y values for plot, here: g is the beginning indicies, jjj[f] is the index value of the cut that did not meet
        #the 30 difference threshold (see jjj)
        x_array = np.arange(0, 5000)
        y_array = np.arange(g,jjj[f])
        xx=[]
        yy=[]
        zz=[]
        for x in x_array:
            for y in y_array:
                #again, appends corresponding x,y and z values from data (dff) in a 1d array accepted by trisurf function
                xx.append(x)
                yy.append(y)
                zz.append(dff[x][y])
        ax.plot_trisurf(xx, yy, zz, cmap='Spectral', edgecolor='none')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
        #g is beginning index, now updated to be the final index + 1
        g=jjj[f]+1


# In[64]:


#same as above in 2d
g=0
for f in range(0,len(jjj)-1):
    if (jjj[f]-jjj[f+1])<-30:
        
        x_array = np.arange(0, 5000)
        y_array = np.arange(g,jjj[f])
        xx=[]
        yy=[]
        zz=[]
        for x in x_array:
            for y in y_array:
                xx.append(x)
                yy.append(y)
                zz.append(dff[x][y])
                #sns heatmap is used to show more detail in the 2d plot
        ax=sns.heatmap(dff[:,g:jjj[f]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plt.show()
        g=jjj[f]+1


# In[92]:


#to clean up the points, I just did an average of every other two points
#setup blank array
dz=[]

#cycling through dataset (dff) to get the average of every other two points, appending twice to get back the same length vector
for i in range(0,5000):
    
    #counting in 2's
    for n in range (0,800,2):
        
        #double appending
        dz.append((dff[i,n]+dff[i,n+1])/2)
        dz.append((dff[i,n]+dff[i,n+1])/2)
        
#setting back to acceptable array form
dz1=np.asarray(dz)

#redoing the same procedure once more, in steps of 4
dzz=[]
for i in range(0,4000000,4):

        #averaging 4 total points, appending 4 times to get same total # of points
        dzz.append((dz1[i] + dz1[i+1] + dz1[i+2] + dz1[i+3]) /4)
        dzz.append((dz1[i] + dz1[i+1] + dz1[i+2] + dz1[i+3]) /4)
        dzz.append((dz1[i] + dz1[i+1] + dz1[i+2] + dz1[i+3]) /4)  
        dzz.append((dz1[i] + dz1[i+1] + dz1[i+2] + dz1[i+3]) /4)

#putting back to array in same format as original dataset (dff)
dzzz = np.array(dzz)
dzzz.resize(5000,800)


# In[98]:


#plotting the cleaned (averaged) version of the dataset
g=0
for f in range(0,len(jjj)-1):
    
    #threshold remains 30, jjj is is still the index array for cutting
    if (jjj[f]-jjj[f+1])<-30:
        
        x_array = np.arange(0, 5000)
        y_array = np.arange(g,jjj[f])
        xx=[]
        yy=[]
        zz=[]
        for x in x_array:
            for y in y_array:
                xx.append(x)
                yy.append(y)
                #this line can be erased as it is not being used but I just left it in,
                zz.append(dff[x][y])
        
        #plotting the cleaned (averaged) version of the dataset named (dzzz) can be renamed if needed.
        ax=sns.heatmap(dzzz[:, g:jjj[f] ])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plt.show()
        g=jjj[f]+1


# In[124]:


for i in range(0,799):
    for j in range (717,1434):
        if (dff[j,i]-dff[j,i+1])<-0.05:
            print(dff[j,i])


# In[133]:


kk=[]
for j in range(0,799):    
        if dff[2000,j]- dff[2000,j+1]>0.004:
            jj.append(j+1)
jj.append(800)

dab=[]
for i in range(0,5000):
    for j in range (0,799):
        if dff[i,j]-dff[i,j+1] > 0:
            dab.append(j)


# In[134]:


dab


# In[148]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(np.arange(0,5000),dff[:,60])
plt.show()


# In[156]:


for i in range(0,4999):
    if np.sqrt((dff[i,60]-dff[i+1,60])*(dff[i,60]-dff[i+1,60]))>1:
        print(i)


# In[171]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(np.arange(0,3602),dff[398:4000,60])
plt.show()


# In[184]:


np.polyfit(np.arange(398,920), dff[398:920,60], 2, rcond=None, full=False)


# In[167]:


dff[398:4000,60]


# In[182]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(np.arange(398,920),dff[398:920,60])
plt.show()


# In[193]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(np.arange(926,950),dff[926:950,63])
plt.show()


# In[ ]:




