"""
Written by Shreya and Phoebe

Program which plots the focus and centroid for a given infile.

Infile should be in the format:
# framenumber,timeSinceStart,focus,blobx,bloby,sigx,sigy,amp,theta,offset

"""

import matplotlib.pyplot as plt
import numpy as np

filename = 'focus.txt'
print 'Reading from '+ filename
# open file, which should be in the format:
# framenumber,timeSinceStart,focus,blobx,bloby,sigx,sigy,amp,theta,offset
with open(filename, 'r') as infile:
    lines = [line.strip().split(',') for line in infile.readlines()]

#turn all the strings into floats
lines = [[float(item) for item in line] for line in lines]

# define function to get indices where each run recorded in focus.txt starts
def getI():
    """returns a list of indices where each run starts"""
    i = 0
    indexs = []
    for line in lines:
        if line[0] == 0.0:
            indexs.append(i)
        i+=1
    return indexs

# call function 
indexs = getI()
indexLength = len(indexs)

# if there's more than one run, take only the last one
if indexLength > 1:
    run1 = lines[indexs[-1]:]
# if there's only one run nothing needs to change
else:
    run1 = lines

# extract data
time = [line[1] for line in run1]
focus = [line[2] for line in run1]
centerx = [line[3] for line in run1]
centery = [line[4] for line in run1]

#calculate centroid
centroid = [np.sqrt(x**2 + y**2) for x, y in zip(centerx, centery)]

#fig1 = plt.figure()
#plt.plot(time, focus,'.-')
#plt.title('Focus vs. Time')
#plt.xlabel('Time (s)')
#plt.ylabel('Focus')
#plt.show()

#fig2 = plt.figure()
#plt.plot(time, centroid,'.-')
#plt.title('Centroid vs. Time')
#plt.xlabel('Time (s)')
#plt.ylabel('Centroid')
#plt.show()

f,axarr = plt.subplots(2,sharex=True)
f.suptitle('Focus, Centroid vs. Time')
axarr[0].plot(time,focus,'.-')
axarr[0].set_ylabel('Focus')
axarr[1].plot(time,centroid,'.-')
axarr[1].set_ylabel('Centroid')
plt.xlabel('Time (s)')
plt.show()
