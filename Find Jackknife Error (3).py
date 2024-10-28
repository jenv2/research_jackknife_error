#!/usr/bin/env python
# coding: utf-8

# In[2]:


import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#Create jackknife bins
def jackknife_bin_creation(values):
    n = len(values)
    jackknife_bins = np.empty((n, n-1))
    jackknife_averages = np.empty(n)
    
    for i in range(n):
        jackknife_bins[i] = np.delete(values, i) #remove a value for each bin
        jackknife_averages[i] = np.mean(jackknife_bins[i]) #calculate average for remaining values
        
    return jackknife_bins, jackknife_averages


# In[4]:


#Extract values and create bins
def extract_values(zip_path, time_slots):    
    extracted_values = [[] for _ in range(time_slots)]  # For storing values from the first 64 rows
    jackknife_bins = [[] for _ in range(time_slots)]    # For storing jackknife bins
    jackknife_averages = [[] for _ in range(time_slots)]
    num_files = 0  # Count the number of files processed  
    
    with zipfile.ZipFile(zip_path, 'r') as z:        
        data_files = [f for f in z.namelist() if f.endswith('.dat') and '__MACOSX' not in f and '1196' not in f] #go into pion data folder, and remove file 1196 since it has NANs
        
        for file_path in data_files: #for each measure (folder) in the overall data file. file_id is 0,1,...,num_files-1
            num_files += 1
            #print(f"Processing file: {file_path}")
            try:
                with z.open(file_path) as file:
                    for i in range(time_slots):
                        line = file.readline().decode('utf-8').strip()
                        if not line:
                            print(f"File {file_path} has less than {time_slots} lines")
                            break;
                        parts = line.split()
                        if len(parts) >= 5:
                            extracted_value = parts[4]
                            try:
                                extracted_values[i].append(float(extracted_value))
                            except ValueError:
                                print(f"Error converting value to float in file {file_path}, line {i+1}: {extracted_value}")
                        else:
                            print(f"Unexpected format in file {file_path}, line {i+1}: {line}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    #now create jackknife bins
    for i in range(time_slots):
        if len(extracted_values[i]) > 1:
            jackknife_bins[i], jackknife_averages[i] = jackknife_bin_creation(np.array(extracted_values[i]))
        else:
            jackknife_bins[i], jackknife_averages[i] = np.array([])

    return extracted_values, jackknife_bins, jackknife_averages, num_files


# In[56]:


root_directory = 'pion_P0 1.zip'
time_slots = 64

extracted_values, jackknife_bins, jackknife_averages, num_files = extract_values(root_directory, time_slots)


# In[17]:


#Folding
def fold_values(jackknife_averages):
    fold_range = 33
    folded_values = np.zeros((fold_range, len(jackknife_averages[0])))
    
    for t in range(fold_range):
        if t==0 or t==32:
            folded_values[t] = jackknife_averages[t]
        else:
            folded_values[t] = (jackknife_averages[t] + jackknife_averages[63-t])/2
                
    return folded_values


# In[57]:


folded_values = fold_values(jackknife_averages)
#len(folded_values)
folded_values


# In[6]:


#Average 1997 values for each time slot to get one time slot average each
def calculate_overall_averages(folded_values):
    averages = np.zeros(len(folded_values))

    for t in range(len(folded_values)):
        averages[t] = np.mean(folded_values[t])
    
    return averages


# In[58]:


folded_averages = calculate_overall_averages(folded_values)

#Plot averages
times = np.arange(0, 33)

plt.errorbar(times, folded_averages, fmt='o', capsize=5)
plt.xlabel('Time slice')
plt.ylabel('Average')
plt.show()


# In[155]:


unfolded_averages = calculate_overall_averages(jackknife_bins)

#Plot unfolded averages
times = np.arange(0, 64)

plt.errorbar(times, unfolded_averages, fmt='o', capsize=5)
plt.xlabel('Time slice')
plt.ylabel('Average')
plt.show()


# In[20]:


#Calculate the log ratios for every t and t+1
def calculate_logs(folded_values):
    i_ratios = np.empty((len(folded_values)-1, len(folded_values[0])))
    ratios = np.empty(len(folded_values)-1) #no ratio for the last t since there's no t+1
    
    for t in range(len(folded_values)-1): #for every time slice except the last one
        for i in range(len(folded_values[t])): #for every bin in t
            this_value = folded_values[t][i]
            other_value = folded_values[t+1][i]
            i_ratios[t][i] = np.log(this_value / other_value)
        ratios[t] = np.mean(i_ratios[t])
    
    return ratios, i_ratios


# In[59]:


logs, bin_logs = calculate_logs(folded_values)
#logs
bin_logs
#len(bin_logs[0]) #1198
#len(bin_logs) #32
#bin_logs[0][1]
len(logs)


# In[234]:


#Plot logs
times = np.arange(0, 32)

plt.errorbar(times, logs, fmt='o', capsize=5)
plt.grid(True)
#plt.axhline(y=0.1253,color='black') #test to see if it plateaus at the correct spot
plt.xlabel('Time slice')
plt.ylabel('Log')
plt.title('Energy Plot (without errors)')
plt.show()


# In[22]:


def calculate_errors(bins):
    errors = np.empty(len(bins)-1)
    
    n = len(bins[0])
    
    for t in range(len(bins)-1):
        std_dev = np.std(bins[t])
        errors[t] = std_dev * np.sqrt(n-1) #error is std_dev * sqrt(n-1)
    
    return errors


# In[60]:


errors = calculate_errors(folded_values)
errors
#len(errors)


# In[237]:


#Plot errors and averages (log ratios)
times = np.arange(0, 32)

plt.errorbar(times, logs, yerr=errors, fmt='o', capsize=5)
plt.grid(True)
plt.xlabel('Time slice')
plt.ylabel('Log')
plt.title('Logs with errors')
plt.show()


# In[337]:


#Plotting Jayden's values with mine
jayden_logs = [0.24019471371222303, 0.16755156192178108, 0.14728899476444, 0.13449068058293898, 0.12866712682533898, 0.12757558917688233, 0.12645208828029364, 0.1261609646275708, 0.12451664719729429, 0.12483373270607748, 0.12374901968357281, 0.1240923299410589, 0.12308434668702564, 0.12326131464411108, 0.12153625846889324, 0.12123143710047661, 0.12020291608487495, 0.11834823229677027, 0.11441764831388902, 0.11204098046999567, 0.11157733715159879, 0.10605017756401498, 0.10166490772836306, 0.09504482259362111, 0.08824844948682545, 0.07981178622460068, 0.06898879595927133, 0.05755221565208663, 0.044123839854234716, 0.029020427335410045, 0.014918553707373314]
jayden_errors =  [1.2985409848201437e-16, 1.330603725186073e-16, 8.737096749715781e-17, 4.007842545741184e-17, 2.003921272870592e-17, 9.61882210977884e-18, 4.5689405021449497e-17, 5.130038458548715e-17, 2.6451760801891813e-17, 5.330430585835774e-17, 3.687215142081889e-17, 4.087999396656008e-17, 9.859292662523313e-17, 8.416469346056486e-18, 4.007842545741184e-19, 2.244391825615063e-17, 1.0300155342554843e-16, 6.011763818611776e-17, 4.488783651230126e-17, 7.775214538737896e-17, 3.4467445893374184e-17, 2.805489782018829e-17, 1.8035291455835328e-17, 3.647136716624477e-17, 6.813332327760013e-17, 6.612940200472953e-17, 3.6070582911670656e-17, 2.3646271019872984e-17, 2.4247447401734164e-17, 8.717057536987074e-18, 5.00980318217648e-20]

new_jayden_logs = np.empty(30)
for i in range(len(new_jayden_logs)):
    new_jayden_logs[i] = jayden_logs[i+1]

new_jayden_errors = np.empty(30)
for i in range(len(new_jayden_errors)):
    new_jayden_errors[i] = jayden_errors[i+1]

len(new_jayden_errors)


# In[340]:


times = np.arange(1, 31)

plt.errorbar(times, logs[1:31], yerr=errors[1:31], fmt='o', capsize=5)
plt.errorbar(times, new_jayden_logs, yerr=new_jayden_errors, fmt='o', capsize=5)
plt.grid(True)
plt.xlabel('Time slice')
plt.ylabel('Log')
plt.title('Logs with errors')
plt.show()


# In[238]:


#Zoom in on log plot (0.05 to 0.2)
times = np.arange(0, 32)

plt.errorbar(times, logs, fmt='o', capsize=5)
plt.ylim(0.05, 0.2)
plt.grid(True)
plt.xlabel('Time slice')
plt.ylabel('Log')
plt.title('Energy Plot, zoomed')
plt.show()


# In[239]:


#Zoom in on log plot (0.05 to 0.15)
times = np.arange(0, 32)

plt.errorbar(times, logs, fmt='o', capsize=5)
plt.ylim(0.05, 0.15)
plt.grid(True)
plt.xlabel('Time slice')
plt.ylabel('Log')
plt.title('Energy Plot, zoomed')
plt.show()


# **X-axis plateau range: 8 to 13**

# In[24]:


def plateau_fit(low, high, errors, bin_logs):
    bin_Es = np.empty(len(bin_logs[0])) #no t dependence
    #avg_Es = np.empty(high-low+1) just a number -> average of bin_Es
    avg_E = 0
    
    for bin in range(len(bin_logs[0])): #for all bins
        nsum = 0
        dsum = 0
        for t in range(low, high+1): #for each time slice in the plateau range
            nsum += bin_logs[t][bin] * (1/(errors[t]**2))
            dsum += (1/errors[t])**2
        
        bin_Es[bin] = nsum/dsum
    
    avg_E = np.mean(bin_Es)
    
    return bin_Es, avg_E


# In[61]:


bin_Es, avg_E = plateau_fit(8, 13, errors, bin_logs)
#bin_Es
avg_E
#len(bin_Es)
#len(bin_Es[1])


# In[62]:


def calculate_E_errors(bin_Es):
    n = len(bin_Es)
        
    std_dev = np.std(bin_Es)
    E_error = std_dev * np.sqrt(n-1)
    
    return E_error


# In[63]:


E_error = calculate_E_errors(bin_Es)
E_error


# **Now, do same for P1**

# In[29]:


root_directory = 'pion_P1 1.zip'
time_slots = 64

extracted_values, jackknife_bins, jackknife_averages, num_files = extract_values(root_directory, time_slots)


# In[32]:


folded_values = fold_values(jackknife_averages)
#len(folded_values)
folded_values


# In[33]:


folded_averages = calculate_overall_averages(folded_values)

#Plot averages
times = np.arange(0, 33)

plt.errorbar(times, folded_averages, fmt='o', capsize=5)
plt.xlabel('Time slice')
plt.ylabel('Average')
plt.show()


# In[37]:


logs, bin_logs = calculate_logs(folded_values)


# In[38]:


#Plot logs
times = np.arange(0, 32)

plt.errorbar(times, logs, fmt='o', capsize=5)
plt.grid(True)
plt.xlabel('Time slice')
plt.ylabel('Log')
plt.title('P1 Energy Plot (without errors)')
plt.show()


# In[40]:


errors = calculate_errors(folded_values)
errors
#len(errors)


# In[41]:


#Plot errors and averages (log ratios)
times = np.arange(0, 32)

plt.errorbar(times, logs, yerr=errors, fmt='o', capsize=5)
plt.grid(True)
plt.xlabel('Time slice')
plt.ylabel('Log')
plt.title('P1 Logs with errors')
plt.show()


# In[43]:


#Zoom in on log plot (0.1 to 0.3)
times = np.arange(0, 32)

plt.errorbar(times, logs, fmt='o', capsize=5)
plt.ylim(0.1, 0.3)
plt.grid(True)
plt.xlabel('Time slice')
plt.ylabel('Log')
plt.title('P1 Energy Plot, zoomed')
plt.show()


# **X-axis plateau range: 10 to 16**

# In[44]:


bin_Es, avg_E = plateau_fit(10, 16, errors, bin_logs)
avg_E


# In[55]:


E_error = calculate_E_errors(bin_Es)
E_error

