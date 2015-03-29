#!/usr/bin/python

import operator
import os
from math import *
import sys
from os.path import isfile
from itertools import *
import numpy as np
import matplotlib.mlab as mlab
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

import pylab as plot
params = {'legend.fontsize': 20,
          'legend.linewidth': 4}
plot.rcParams.update(params)

machePath = "Output_Mache/"
sanderPath = "Output_Sander/"

dirs = [x for x in os.listdir(machePath) if os.path.isdir(sanderPath+x) and os.path.isdir(sanderPath+x)]

print dirs
fig = plt.figure()
for d in dirs:
    for f in os.listdir(machePath+d):
      if f.endswith(".out") and f.startswith("time"):
	print f
	inputFile = open(machePath + d + "/" + f)
	steps = []      
	time = []
	
	for l in inputFile:
	  steps.append(int(l.split()[0]))
	  time.append(float(l.split()[1]))
	inputFile.close()
	
	np_time = np.array(time)
	np_step = np.array(steps)
	p1 = plt.plot(np_step, np_time, lw=2, label="mch_"+f)
    
    for f in os.listdir(sanderPath+d):
      if f.endswith(".out") and f.startswith("time"):
	print f
	inputFile = open(sanderPath + d + "/" + f)
	steps = []      
	time = []
	
	for l in inputFile:
	  stp = int(l.split()[0])
	  steps.append(stp)
	  time.append(float(l.split()[1]))		#Time is in S and is an average
	inputFile.close()
	
	np_time = np.array(time)
	np_step = np.array(steps)
	p1 = plt.plot(np_step, np_time, lw=2, label="snd_"+f)

    legend=plt.legend(bbox_to_anchor=(0.38, 0.52), loc=4, borderaxespad=0.,
			fancybox=True,shadow=True,title='Method')
    plt.grid()
    plt.show()
    fig.savefig(machePath+d+'/out.png', dpi = 100)