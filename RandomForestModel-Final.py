#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import scipy
from scipy import stats as st


# In[2]:


data=pd.read_csv('Final_Dataset_3.csv')
print(data.head())


# In[3]:


feature_data_wateractual= data.drop(["WaterActual","DoughMoisture","Temperature","WetWeight","Zone1","Zone2","Zone3","Zone4","SteamDamper1","SteamDamper2","SteamDamper3","SteamDamper4"],axis=1)
feature_data_wateractual = np.array(feature_data_wateractual)

feature_data_zone1= data.drop(["MOISTURE","BiscuitPasteActual","DoughMoisture","Temperature","WetWeight","Zone1","Zone2","Zone3","Zone4","SteamDamper1","SteamDamper2","SteamDamper3","SteamDamper4"],axis=1)
feature_data_zone1 = np.array(feature_data_zone1)
feature_data_zone2= data.drop(["MOISTURE","BiscuitPasteActual","Temperature","WetWeight","Zone2","Zone3","Zone4","SteamDamper1","SteamDamper2","SteamDamper3","SteamDamper4"],axis=1)
feature_data_zone2 = np.array(feature_data_zone2)
feature_data_zone3= data.drop(["MOISTURE","BiscuitPasteActual","DoughMoisture","Temperature","WetWeight","Zone1","Zone3","Zone4","SteamDamper1","SteamDamper2","SteamDamper3","SteamDamper4"],axis=1)
feature_data_zone3 = np.array(feature_data_zone3)
feature_data_zone4= data.drop(["BiscuitPasteActual","DoughMoisture","Temperature","WetWeight","Zone1","Zone4","SteamDamper1","SteamDamper2","SteamDamper3","SteamDamper4"],axis=1)
feature_data_zone4 = np.array(feature_data_zone4)

feature_data_steamdamper1= data.drop(["MOISTURE","BiscuitPasteActual","DoughMoisture","Temperature","WetWeight","Zone1","Zone2","Zone3","Zone4","SteamDamper1","SteamDamper2","SteamDamper3","SteamDamper4"],axis=1)
feature_data_steamdamper1 = np.array(feature_data_steamdamper1)
feature_data_steamdamper2= data.drop(["BiscuitPasteActual","Temperature","WetWeight","Zone1","Zone2","Zone3","Zone4","SteamDamper1","SteamDamper2","SteamDamper3","SteamDamper4"],axis=1)
feature_data_steamdamper2 = np.array(feature_data_steamdamper2)
feature_data_steamdamper3= data.drop(["BiscuitPasteActual","Temperature","WetWeight","Zone1","Zone2","Zone3","Zone4","SteamDamper3","SteamDamper4"],axis=1)
feature_data_steamdamper3 = np.array(feature_data_steamdamper3)
feature_data_steamdamper4= data.drop(["BiscuitPasteActual","DoughMoisture","Temperature","WetWeight","Zone1","Zone2","Zone3","Zone4","SteamDamper1","SteamDamper4"],axis=1)
feature_data_steamdamper4 = np.array(feature_data_steamdamper4)


# In[4]:


labels_wateractual = np.array(data['WaterActual'])

labels_zone1 = np.array(data['Zone1'])
labels_zone2 = np.array(data['Zone2'])
labels_zone3 = np.array(data['Zone3'])
labels_zone4 = np.array(data['Zone4'])

labels_steamdamper1 = np.array(data['SteamDamper1'])
labels_steamdamper2 = np.array(data['SteamDamper2'])
labels_steamdamper3 = np.array(data['SteamDamper3'])
labels_steamdamper4 = np.array(data['SteamDamper4'])


# In[5]:


train_feature_data_wateractual, test_feature_data_wateractual, train_labels_wateractual, test_labels_wateractual = train_test_split(feature_data_wateractual, labels_wateractual, test_size = 0.25, random_state = 42)

train_feature_data_zone1, test_feature_data_zone1, train_labels_zone1, test_labels_zone1 = train_test_split(feature_data_zone1, labels_zone1, test_size = 0.25, random_state = 42)
train_feature_data_zone2, test_feature_data_zone2, train_labels_zone2, test_labels_zone2 = train_test_split(feature_data_zone2, labels_zone2, test_size = 0.25, random_state = 42)
train_feature_data_zone3, test_feature_data_zone3, train_labels_zone3, test_labels_zone3 = train_test_split(feature_data_zone3, labels_zone3, test_size = 0.25, random_state = 42)
train_feature_data_zone4, test_feature_data_zone4, train_labels_zone4, test_labels_zone4 = train_test_split(feature_data_zone4, labels_zone4, test_size = 0.25, random_state = 42)

train_feature_data_steamdamper1, test_feature_data_steamdamper1, train_labels_steamdamper1, test_labels_steamdamper1 = train_test_split(feature_data_steamdamper1, labels_steamdamper1, test_size = 0.25, random_state = 42)
train_feature_data_steamdamper2, test_feature_data_steamdamper2, train_labels_steamdamper2, test_labels_steamdamper2 = train_test_split(feature_data_steamdamper2, labels_steamdamper2, test_size = 0.25, random_state = 42)
train_feature_data_steamdamper3, test_feature_data_steamdamper3, train_labels_steamdamper3, test_labels_steamdamper3 = train_test_split(feature_data_steamdamper3, labels_steamdamper3, test_size = 0.25, random_state = 42)
train_feature_data_steamdamper4, test_feature_data_steamdamper4, train_labels_steamdamper4, test_labels_steamdamper4 = train_test_split(feature_data_steamdamper4, labels_steamdamper4, test_size = 0.25, random_state = 42)


# In[6]:


from sklearn.ensemble import RandomForestRegressor

Random_Forest_WaterActual = RandomForestRegressor(n_estimators = 25, random_state = 42)
Random_Forest_WaterActual.fit(train_feature_data_wateractual, train_labels_wateractual);

Random_Forest_Zone1 = RandomForestRegressor(n_estimators = 25, random_state = 42)
Random_Forest_Zone1.fit(train_feature_data_zone1, train_labels_zone1);

Random_Forest_Zone2 = RandomForestRegressor(n_estimators = 25, random_state = 42)
Random_Forest_Zone2.fit(train_feature_data_zone2, train_labels_zone2);

Random_Forest_Zone3 = RandomForestRegressor(n_estimators = 15, random_state = 42)
Random_Forest_Zone3.fit(train_feature_data_zone3, train_labels_zone3);

Random_Forest_Zone4 = RandomForestRegressor(n_estimators = 75, random_state = 42)
Random_Forest_Zone4.fit(train_feature_data_zone4, train_labels_zone4);

Random_Forest_SteamDamper1 = RandomForestRegressor(n_estimators = 400, random_state = 42)
Random_Forest_SteamDamper1.fit(train_feature_data_steamdamper1, train_labels_steamdamper1);

Random_Forest_SteamDamper2 = RandomForestRegressor(n_estimators = 400, random_state = 42)
Random_Forest_SteamDamper2.fit(train_feature_data_steamdamper2, train_labels_steamdamper2);

Random_Forest_SteamDamper3 = RandomForestRegressor(n_estimators = 25, random_state = 42)
Random_Forest_SteamDamper3.fit(train_feature_data_steamdamper3, train_labels_steamdamper3);

Random_Forest_SteamDamper4 = RandomForestRegressor(n_estimators = 25, random_state = 42)
Random_Forest_SteamDamper4.fit(train_feature_data_steamdamper4, train_labels_steamdamper4);


# In[7]:


while True:
    try:
        Gluten1 = float(input("Input Gluten1: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")

while True:
    try:
        Gluten2 = float(input("Input Gluten2: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")

while True:
    try:
        SV1 = float(input("Input SV1: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")

while True:
    try:
        SV2 = float(input("Input SV2: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")
        
while True:
    try:
        AshContent1 = float(input("Input AshContent1: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")

while True:
    try:
        AshContent2 = float(input("Input AshContent2: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")
        
        
while True:
    try:
        AIA1 = float(input("Input AIA1: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")

while True:
    try:
        AIA2 = float(input("Input AIA2: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")
        
while True:
    try:
        BrokenStarch1 = float(input("Input BrokenStarch1: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")

while True:
    try:
        BrokenStarch2 = float(input("Input BrokenStarch2: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")

while True:
    try:
        Moisture1 = float(input("Input Moisture1: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")

while True:
    try:
        Moisture2 = float(input("Input Moisture2: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")
        
while True:
    try:
        Granlarity1 = float(input("Input Granlarity1: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")

while True:
    try:
        Granlarity2 = float(input("Input Granlarity2: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")
        
        
while True:
    try:
        BiscuitPasteActual = float(input("Input BiscuitPasteActual: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")

while True:
    try:
        ABCActual = float(input("Input ABCActual: "))
        break
    except ValueError:
        print("\nThis is not a number. Try again...")
               
GLUTEN = 0.5*(Gluten1+Gluten2)
SV = 0.5*(SV1+SV2)
ASHCONTENT = 0.5*(AshContent1+AshContent2)
AIA = 0.5*(AIA1+AIA2)
BrokenStarch = 0.5*(BrokenStarch1+BrokenStarch2)
MOISTURE = 0.5*(Moisture1+Moisture2)
GRANLARITY = 0.5*(Granlarity1+Granlarity2)
#BiscuitPasteActual = BiscuitPasteActual
#ABCActual = ABCActual



# In[8]:


Input_WaterActual=np.array([[GLUTEN,SV,ASHCONTENT,AIA,BrokenStarch,MOISTURE,GRANLARITY,BiscuitPasteActual,ABCActual]])
Final_Prediction_WaterActual= Random_Forest_WaterActual.predict(Input_WaterActual)
WaterActual=Final_Prediction_WaterActual[0]

DoughMoisture=(MOISTURE*1.48 + 45/195*BiscuitPasteActual + WaterActual +0)/ (148+ BiscuitPasteActual + WaterActual + ABCActual)

Input_Zone1=np.array([[GLUTEN,SV,ASHCONTENT,AIA,BrokenStarch,GRANLARITY,ABCActual,WaterActual]])
Final_Prediction_Zone1= Random_Forest_Zone1.predict(Input_Zone1)
Zone1=Final_Prediction_Zone1[0]

Input_Zone2=np.array([[GLUTEN,SV,ASHCONTENT,AIA,BrokenStarch,GRANLARITY,ABCActual,WaterActual,DoughMoisture,Zone1]]) 
Final_Prediction_Zone2= Random_Forest_Zone2.predict(Input_Zone2)
Zone2=Final_Prediction_Zone2[0]

Input_Zone3=np.array([[GLUTEN,SV,ASHCONTENT,AIA,BrokenStarch,GRANLARITY,ABCActual,WaterActual,Zone2]])
Final_Prediction_Zone3= Random_Forest_Zone3.predict(Input_Zone3)
Zone3=Final_Prediction_Zone3[0]

Input_Zone4=np.array([[GLUTEN,SV,ASHCONTENT,AIA,BrokenStarch,MOISTURE,GRANLARITY,ABCActual,WaterActual,Zone2,Zone3]]) 
Final_Prediction_Zone4= Random_Forest_Zone4.predict(Input_Zone4)
Zone4=Final_Prediction_Zone4[0]

Input_SteamDamper1=np.array([[GLUTEN,SV,ASHCONTENT,AIA,BrokenStarch,GRANLARITY,ABCActual,WaterActual]])
Final_Prediction_SteamDamper1= Random_Forest_SteamDamper1.predict(Input_SteamDamper1)
SteamDamper1=10*round(Final_Prediction_SteamDamper1[0]/10)

Input_SteamDamper2=np.array([[GLUTEN,SV,ASHCONTENT,AIA,BrokenStarch,MOISTURE,GRANLARITY,ABCActual,WaterActual,DoughMoisture]])
Final_Prediction_SteamDamper2= Random_Forest_SteamDamper2.predict(Input_SteamDamper2)
SteamDamper2=10*round(Final_Prediction_SteamDamper2[0]/10)

Input_SteamDamper3=np.array([[GLUTEN,SV,ASHCONTENT,AIA,BrokenStarch,MOISTURE,GRANLARITY,ABCActual,WaterActual,DoughMoisture,SteamDamper1,SteamDamper2]])
Final_Prediction_SteamDamper3= Random_Forest_SteamDamper3.predict(Input_SteamDamper3)
SteamDamper3=10*round(Final_Prediction_SteamDamper3[0]/10)

Input_SteamDamper4=np.array([[GLUTEN,SV,ASHCONTENT,AIA,BrokenStarch,MOISTURE,GRANLARITY,ABCActual,WaterActual,SteamDamper2,SteamDamper3]])
Final_Prediction_SteamDamper4= Random_Forest_SteamDamper4.predict(Input_SteamDamper4)
SteamDamper4=10*round(Final_Prediction_SteamDamper4[0]/10)


# In[10]:


print("WaterActual:",round(WaterActual, 2),"\nDoughMoisture:",round(DoughMoisture*100,2),"%","\nZone1:",round(Zone1,2))
print("Zone2:",round(Zone2,2),"\nZone3:",round(Zone3,2),"\nZone4:",round(Zone4,2),"\nSteamDamper1:",SteamDamper1)
print("SteamDamper2:",SteamDamper2,"\nSteamDamper3:",SteamDamper3,"\nSteamDamper4:",SteamDamper4)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




