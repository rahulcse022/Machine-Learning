#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10,4)
plt.style.use('seaborn-whitegrid')


# In[2]:


path = "u.data"
col_names = [ "user_id","movie_id","ratings" ]
ratings = pd.read_csv(path,sep="\t",names=col_names,usecols=[0,1,2])
ratings.head()


# ## EDA 

# In[3]:


plt.hist(ratings['movie_id'],edgecolor='black',color='green',bins=30)
plt.show()


# ### most popular moives 

# In[4]:


ratings['movie_id'].value_counts()[:20].plot(kind='bar')
plt.show()


# In[5]:


ratings.shape


# ##### let's find popularity of movies 
#     
#     based how many pepole watched a movie and average rating of that movie  

# In[6]:


ratings.head()


# In[7]:


movie_property = ratings.groupby("movie_id").agg( 
    {'ratings':np.mean,'user_id':np.size} )
movie_property.columns  = [ 'mean',"size"]


# In[8]:


movie_property.head()


# #### top 5 popular moives 

# In[9]:


movie_property.columns


# In[10]:


movie_property.sort_values('size',ascending=False)[:5]


# $$ \text {normalizing coeff }x_i = \frac { (x_i - min(X)) } { max(X) - min(X) } $$

#     X    popularity 
#     10    ( 10 - 5 ) / ( 15 - 5 ) -->  0.5
#     6     ( 6 - 5 ) / ( 15 - 5 )  -->  0.1
#     15    ( 15 - 5 ) / ( 15 - 5 ) -->  1.0
#     5      ( 5 - 5 ) / ( 15 - 5 ) --> 0 
# 
#     min(X)= 5
#     max(X) = 15 
# 

# In[11]:


movie_property.head()


# In[12]:


minimum = np.min(movie_property['size'])
maximum = np.max(movie_property['size'])
movie_property['popularity']  = movie_property['size'].apply(lambda x: (x-minimum)/(maximum-minimum))
print(" Minimum is ",minimum,"\n","Maximum is ", maximum)


# In[13]:


movie_property.head()


# In[14]:


movie_property.iloc[814]


# In[15]:


columns = [  "movie_id","movie_title","release_date","video_release date",
    "IMDb_URL","unknown","Action","Adventure","Animation",
              "Childrens" , "Comedy","Crime", "Documentary","Drama","Fantasy",
              "Film-Noir_Horror", "Musical" , "Mystery", "Romance", "Sci-Fi",
              "Thriller","War","Western" ,"extra" ] 


# In[16]:


movie_df = pd.read_csv("u.item",encoding='latin',sep="|",
                      names=columns)


# In[17]:


movie_df.head()


# In[18]:


movie_name = movie_df['movie_title']


# In[19]:


movie_name.index = movie_df['movie_id']


# In[20]:


movie_name.head()


# In[21]:


movie_name[50]


# In[22]:


movie_df.head()


# In[23]:


movie_df = movie_df.pivot_table(index='movie_id')


# In[24]:


movie_df.head()


# In[25]:


movie_df.columns


# In[26]:


movie_df.loc[50]


# In[27]:


movie_property.head()


# In[28]:


all_properties = movie_property.join(movie_df)


# In[29]:


all_properties.head()


# In[30]:


all_properties.drop("size",axis=1,inplace=True)


# In[31]:


movie_name.head()


# In[32]:


all_properties.head()


# In[33]:


all_properties.loc[133,'popularity']


# In[34]:


all_properties.loc[50,'popularity']


# In[35]:


movie_name[133]


# In[36]:


# star wars 
# 


# In[37]:


v1 = [ 1,2,3, ] 
v2 = [ 3,2,1 ]


#     Euclidean Distance 
#     Manhatten Distance
#     Hamming Distance
#     Minkowski Distance

# $$ \text {Euclidean Distance (x1,y1) and (x2,y2)} = \sqrt { (x_1 - x_2)^2 + (y_1 - y2)^2}   $$

# $$ Cosine Distance =\frac { \sum\limits_{i=1}^N x_i.y_i }  { \sqrt { \sum\limits_{i=1}^n x_i^2 }  \sqrt { \sum\limits_{i=1}^n y_i^2 } }$$

# In[38]:


from scipy.spatial.distance import cosine


# In[39]:


cosine([0,1,1,1,0,1],[0,0,1,1,0,1])


# In[40]:


all_properties.head(2)


# In[41]:


all_properties.loc[1,'Action':'unknown']


# In[42]:


def calculateDistance(id1,id2):
    pop_dis = abs(all_properties.loc[id1,'popularity'] - all_properties.loc[id2,'popularity'])
    genre_dis = cosine(all_properties.loc[id1,'Action':'unknown'],
                      all_properties.loc[id2,'Action':'unknown']) 
    total_distance = pop_dis + genre_dis
    return total_distance
    
    


# In[43]:


import operator
def getNeighbours(movie_id,k=5):
    distance = [ (mid,calculateDistance(movie_id,mid) )  for mid in all_properties.index if mid != movie_id ] 
    distance.sort(key=operator.itemgetter(1))
    #distance.sort(key=itemge)
    distance = distance[:k]
    return [ mid for mid,dis in distance ]    


# In[44]:


all_properties.loc[50,'popularity']


# In[45]:


def similar_movie(movie_id,k=5) : 
    ids = getNeighbours(movie_id,k)
    print(f"So {k} Nearest Neighbours of {movie_name[movie_id]}\n")
    print("Rating\t\tName")
    for mid in ids : 
        name = movie_name[mid]
        ratings = all_properties.loc[mid,'mean']
        print(f"{ratings:5.2}\t{name}")


# In[46]:


similar_movie(50,10)


# In[47]:


getNeighbours(50)


# In[48]:


from sklearn.preprocessing import StandardScaler


# In[49]:


scale = StandardScaler()
data = scale.fit_transform(all_properties[['mean','popularity']])
data[:5]


# In[50]:


y = all_properties.copy()


# In[51]:


all_properties[['mean','popularity']] =  data


# In[52]:


all_properties.head()


# In[53]:


from sklearn.neighbors import NearestNeighbors


# In[54]:


model = NearestNeighbors(n_neighbors=10)


# In[55]:


model.fit(all_properties.iloc[:,2:])


# In[56]:


all_properties.iloc[50:52]


# In[57]:


data = model.kneighbors(X=all_properties.iloc[50:52,2:],n_neighbors=10)


# In[58]:


data[0].shape


# In[59]:


data[1].shape


# In[60]:


data[1]


# In[61]:


print("Movie Name = ",movie_name.loc[50])  #['movie_title']
print("Nearest Neighbours : \n\t",end='')
print(*movie_name.loc[data[1][0]],sep="\n\t")


# In[62]:


movie_name[movie_name == 'Star Wars (1977)']


# In[63]:


scale= StandardScaler()


# In[64]:


data = scale.fit_transform(all_properties)


# In[65]:


model = NearestNeighbors() 


# In[66]:


model.fit(data)


# In[67]:


data = model.kneighbors(X=all_properties.iloc[50:52,:],n_neighbors=10)


# In[68]:


print("Movie Name = ",movie_name.loc[50])
print("Nearest Neighbours : \n\t",end='')
print(*movie_name.loc[data[1][0]],sep="\n\t")


# In[ ]:




