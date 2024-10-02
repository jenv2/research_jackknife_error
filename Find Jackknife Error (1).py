#!/usr/bin/env python
# coding: utf-8

# In[3]:


import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[235]:


#Read the 5th column of the first 64 rows from a data file
def process_data_file(file):
    df = pd.read_csv(file, delim_whitespace=True)
    
    all_data = df.iloc[:64, 4].values #select first 64 rows and 5th column
    
    m = len(all_data) #equals 64
    n = m//2 #equals 32

    folded_data = np.empty(n+1) #length of 33 (index 0 to 32)
    
    folded_data[0] = all_data[0] #keep t=0 the same
    
    for i in range(1, n):
        #print(f"Folding: i={i}, m-i={m-i}, all_data[i]={all_data[i]}, all_data[m-i]={all_data[m-i]}")
        folded_data[i] = (all_data[i]+all_data[m-i]) / 2
        
    folded_data[n] = all_data[n] #keep t=32 the same
    
    return folded_data


# In[213]:


#Create jackknife bins
def jackknife_bin_creation(data):
    n = len(data)
    jackknife_bins = np.empty((n, n-1))
    jackknife_averages = np.empty(n)
    
    for i in range(n):
        jackknife_bins[i] = np.delete(data, i) #remove a value for each bin
        jackknife_averages[i] = np.mean(jackknife_bins[i]) #calculate average for remaining values (excluding the 1)
        
    return jackknife_averages


# In[236]:


#Main processing function
def process_zipfile(zip_path, num_files):    
    all_bins = np.zeros((33, num_files)) #we want 33 sets of bins##########
            
    with zipfile.ZipFile(zip_path, 'r') as z:        
        data_files = [f for f in z.namelist() if 'pion_P0' in f and f.endswith('.dat') and '__MACOSX' not in f and '1196' not in f] #go into pion data folder, and remove file 1196 since it has NANs
        
        for file_id, measure in enumerate(data_files): #for each measure (folder) in the overall data file. file_id is 0,1,...,num_files-1
            with z.open(measure) as file:
                try:
                    #print(f"MEASURE: {measure}")
                    data = process_data_file(file) #get the 5th columns of the first 64 rows, then folded
                    
                    for i in range(len(data)):
                        all_bins[i, file_id] = data[i] #have multiple values for each i from all the different measures. an array of arrays
                except Exception as e:
                    print(f"Error processing file {measure}: {e}")
    
    final_all_bins = []
    for rows in all_bins:
        final_all_bins.append(jackknife_bin_creation(np.array(rows))) #make bins for each time slice
    
    return final_all_bins


# In[217]:


zip_file_path = 'pion_P0 1.zip' #momentum = 0

num_files = 1198
jackknife_bins = process_zipfile(zip_file_path, int(num_files))


# In[237]:


#WITHOUT FOLDING
jackknife_bins_no_fold = process_zipfile(zip_file_path, int(num_files))


# In[238]:


jackknife_bins_no_fold[0][1]


# In[239]:


jackknife_bins_no_fold[1][1]


# In[240]:


jackknife_bins_no_fold[2][1]


# In[218]:


jackknife_bins[0][1]


# In[219]:


jackknife_bins[1][1] #<--- we see an issue here. It should not be a value larger than at t=2


# In[220]:


jackknife_bins[2][1]


# In[221]:


len(jackknife_bins[0])


# In[222]:


len(jackknife_bins)


# In[ ]:


for i in len(jackknife_bins[63]) {
    print(jackknife_bins[63][i])
}


# In[224]:


n = len(jackknife_bins)
averages1 = np.empty(n)
this = np.empty(len(jackknife_bins[0]))

for t in range(n):
    for i in range(len(jackknife_bins[t])):
        this[i] = jackknife_bins[t][i]
    averages1[t] = np.mean(this)


# In[225]:


averages1


# In[226]:


len(averages1)


# In[228]:


#Plot bin averages
times = np.arange(0, 33)

plt.errorbar(times, averages1, fmt='o', capsize=5)
plt.xlabel('Time slice')
plt.ylabel('Effective mass')
plt.show()


# In[189]:


#Plot bin averages (before folding)
times = np.arange(0, 64)###

plt.errorbar(times, averages1, fmt='o', capsize=5)
plt.xlabel('Time slice')
plt.ylabel('Effective mass')
plt.show()


# In[229]:


#Calculate the log ratios for every t and t+1
def calculate_ratios(jackknife_bins):
    i_ratios = np.empty(len(jackknife_bins[0])) #size 1199
    ratios = np.empty(len(jackknife_bins)-1) #size 32. no ratio for the last t since there's no t+1
    
    for t in range(len(jackknife_bins)-1): #for every time slice except the last one
        for i in range(len(jackknife_bins[t])): #for every bin in t
            this_value = jackknife_bins[t][i]
            other_value = jackknife_bins[t+1][i]
            i_ratios[i] = np.log(this_value / other_value)
        ratios[t] = np.mean(i_ratios)
    
    return ratios


# In[230]:


ratios = calculate_ratios(jackknife_bins) #these are the effective masses to plot


# In[231]:


print(ratios)


# In[232]:


#Calculate error for every t
def calculate_error(jackknife_bins):
    errors = np.empty(len(jackknife_bins)-1) #32 errors

    for t in range(len(jackknife_bins)-1):
        part_two = np.sqrt((len(jackknife_bins[t]) - 1) / len(jackknife_bins[t]))
    
        differences = np.empty(len(jackknife_bins[t]))
        for i in range(len(jackknife_bins[t])):
            differences[i] = (jackknife_bins[t][i] - ratios[t])**2
        part_one = np.sqrt(np.sum(differences))
    
        errors[t] = part_one * part_two
    return errors


# In[245]:


errors = calculate_error(jackknife_bins)
errors


# In[244]:


#Plot errors and averages (log ratios)
times = np.arange(0, len(jackknife_bins)-1)

plt.errorbar(times, ratios, fmt='o', capsize=5)
plt.xlabel('Time slice')
plt.ylabel('Effective mass')
plt.title('Effective masses with errors')
plt.show()

