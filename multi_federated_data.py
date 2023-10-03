import pandas as pd
import os
from collections import defaultdict
import pickle


import re

reg="\d{1,}[^a-zA-Z]?\d{1,}?[^a-zA-Z]?\d?"


mutlti_federated_data=[]





#data directory path
path='/var/share/rs1/LCL_DATA/preparded_houshold_data'

#getting the names of all files of users
file_names=os.listdir(path)

print(file_names)

# dictionary for containing the user with different electricity meter types
dic_Std_ACORN_A=defaultdict(list)
dic_Std_ACORN_B=defaultdict(list)
dic_Std_ACORN_C=defaultdict(list)
dic_Std_ACORN_D=defaultdict(list)
dic_Std_ACORN_E=defaultdict(list)
dic_Std_ACORN_F=defaultdict(list)
dic_Std_ACORN_G=defaultdict(list)
dic_Std_ACORN_H=defaultdict(list)
dic_Std_ACORN_I=defaultdict(list)
dic_Std_ACORN_J=defaultdict(list)
dic_Std_ACORN_K=defaultdict(list)
dic_Std_ACORN_L=defaultdict(list)
dic_Std_ACORN_M=defaultdict(list)
dic_Std_ACORN_N=defaultdict(list)
dic_Std_ACORN_O=defaultdict(list)
dic_Std_ACORN_P=defaultdict(list)
dic_Std_ACORN_Q=defaultdict(list)
dic_Std_ACORN_U=defaultdict(list)



dic_ToU_ACORN_A=defaultdict(list)
dic_ToU_ACORN_B=defaultdict(list)
dic_ToU_ACORN_C=defaultdict(list)
dic_ToU_ACORN_D=defaultdict(list)
dic_ToU_ACORN_E=defaultdict(list)
dic_ToU_ACORN_F=defaultdict(list)
dic_ToU_ACORN_G=defaultdict(list)
dic_ToU_ACORN_H=defaultdict(list)
dic_ToU_ACORN_I=defaultdict(list)
dic_ToU_ACORN_J=defaultdict(list)
dic_ToU_ACORN_K=defaultdict(list)
dic_ToU_ACORN_L=defaultdict(list)
dic_ToU_ACORN_M=defaultdict(list)
dic_ToU_ACORN_N=defaultdict(list)
dic_ToU_ACORN_O=defaultdict(list)
dic_ToU_ACORN_P=defaultdict(list)
dic_ToU_ACORN_Q=defaultdict(list)
dic_ToU_ACORN_U=defaultdict(list)



i=0
for file_name in file_names:
    
    result = re.search('block(.*)_MAC(.*)', file_name)
    #print(result.group(1))
    
    block_num=int(result.group(1))
    
  
  
    
    # #get the block number 
    # block_num=int(file_name[5])
    
    #condition on number of blocks
    if block_num < 5:
        
        #read the user info as pandas dataframe 
        df=pd.read_csv(os.path.join(path,file_name))
        
        
       
        
       
        if df['stdorToU'][1]=='Std':
            
            if df['Acorn'][1]=='ACORN-A':
                dic_Std_ACORN_A[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-B':
                dic_Std_ACORN_B[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-C':
                dic_Std_ACORN_C[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-D':
                dic_Std_ACORN_D[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-E':
                dic_Std_ACORN_E[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-F':
                dic_Std_ACORN_F[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-G':
                dic_Std_ACORN_G[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-H':
                dic_Std_ACORN_H[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-I':
                dic_Std_ACORN_I[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-J':
                dic_Std_ACORN_J[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
                
            elif df['Acorn'][1]=='ACORN-K':
                dic_Std_ACORN_K[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-L':
                dic_Std_ACORN_L[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-M':
                dic_Std_ACORN_M[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn_grouped'][1]=='ACORN-N':
                dic_Std_ACORN_N[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-O':
                dic_Std_ACORN_O[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-P':
                dic_Std_ACORN_P[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-Q':
                dic_Std_ACORN_Q[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-U':
                dic_Std_ACORN_U[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
                
                
                
                
                
                
        elif df['stdorToU'][1]=='ToU':
            
            if df['Acorn'][1]=='ACORN-A':
                dic_ToU_ACORN_A[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-B':
                dic_ToU_ACORN_B[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-C':
                dic_ToU_ACORN_C[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-D':
                dic_ToU_ACORN_D[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-E':
                dic_ToU_ACORN_E[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-F':
                dic_ToU_ACORN_F[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-G':
                dic_ToU_ACORN_G[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-H':
                dic_ToU_ACORN_H[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-I':
                dic_ToU_ACORN_I[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-J':
                dic_ToU_ACORN_J[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-K':
                dic_ToU_ACORN_K[block_num].append(df['energy(kWh/hh)'])
                print('5555555')
            elif df['Acorn'][1]=='ACORN-L':
                dic_ToU_ACORN_L[block_num].append(df['energy(kWh/hh)'])
            elif df['Acorn'][1]=='ACORN-M':
                dic_ToU_ACORN_M[block_num].append(df['energy(kWh/hh)'])
            elif df['Acorn'][1]=='ACORN-N':
                dic_ToU_ACORN_N[block_num].append(df['energy(kWh/hh)'])
            elif df['Acorn'][1]=='ACORN-O':
                dic_ToU_ACORN_O[block_num].append(df['energy(kWh/hh)'])
            elif df['Acorn'][1]=='ACORN-P':
                dic_ToU_ACORN_P[block_num].append(df['energy(kWh/hh)'])
            elif df['Acorn'][1]=='ACORN-Q':
                dic_ToU_ACORN_Q[block_num].append(df['energy(kWh/hh)'])
            elif df['Acorn'][1]=='ACORN-U':
                dic_ToU_ACORN_U[block_num].append(df['energy(kWh/hh)'])
            print('block'+str(block_num)+ df['LCLid'][5]+'is ToU')

#list for all 
list_dics_std=[dic_Std_ACORN_A,dic_Std_ACORN_B,dic_Std_ACORN_C,dic_Std_ACORN_D,dic_Std_ACORN_E,dic_Std_ACORN_F,dic_Std_ACORN_G,
               dic_Std_ACORN_H,dic_Std_ACORN_I,dic_Std_ACORN_J,dic_Std_ACORN_K,dic_Std_ACORN_L,dic_Std_ACORN_M,dic_Std_ACORN_N,
               dic_Std_ACORN_O,dic_Std_ACORN_P,dic_Std_ACORN_Q,dic_Std_ACORN_U]

list_dics_ToU=[dic_ToU_ACORN_A,dic_ToU_ACORN_B,dic_ToU_ACORN_C,dic_ToU_ACORN_D,dic_ToU_ACORN_E,dic_ToU_ACORN_F,dic_ToU_ACORN_G,
               dic_ToU_ACORN_H,dic_ToU_ACORN_I,dic_ToU_ACORN_J,dic_ToU_ACORN_K,dic_ToU_ACORN_L,dic_ToU_ACORN_M,dic_ToU_ACORN_N,
               dic_ToU_ACORN_O,dic_ToU_ACORN_P,dic_ToU_ACORN_Q,dic_ToU_ACORN_U]


for item in list_dics_std:
    print(len(item[3]))
    
for item in list_dics_ToU:
    print(len(item[3]))      
    
 
list_of_clustered=[]

for item in list_dics_std:
    
    for i in range (0,block_num):
        
        if len(item[i])!=0:
            list_of_clustered.append(item[i])
        

for item in list_dics_ToU:
    
    for i in range (0,block_num):
        
        if len(item[i])!=0:
            list_of_clustered.append(item[i])
    

for item in list_of_clustered:
    print(len(item))
    

with open("list_of_clusters", "wb") as fp:
    pickle.dump(list_of_clustered, fp)

with open("list_of_clusters", "rb") as fp:   # Unpickling
     read_list = pickle.load(fp)      
print(len(read_list))
