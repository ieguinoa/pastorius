#!/usr/bin/python

import operator
import os
from math import *
import sys
from os.path import isfile
from itertools import *
#import numpy as np
import string
from datetime import datetime

#***********************************************************************
#************************** Global Variables ***************************
#***********************************************************************

endl = "\n"
tab = "\t"
make = False
time = False
coordenates = False
derivativeMode = False
analyticMode = True
periodicity = False
cpu=False
texture=False
debug = True
textureSize=False
texSize=65000
currScnd= datetime.now().second
currMin = datetime.now().minute
currHour = datetime.now().hour


#************************
#Folders and files names:
#************************
today = "_"+str(currHour)+"_"+str(currMin)+"_"+str(currScnd)
defaultTopFile = "input.prmtop"
defaultRstFile = "input.rst7"
defaultMdinFile = "input.mdin"

input_amber_path = "Input_Amber/"
output_path = "Input_Mache/"
output_sander_path = "Output_Sander/"
output_Mache_path = "Output_Mache/"
lennardTableFilename = "TablaLennardOriginal"
lennardOutFilename = "TablaCoeficientesLennard"
particlesFilename = "particles.in"
outSanderFilename = "outSander"
output_Mache_Times_path = output_Mache_path
output_sander_Times_path = output_sander_path
timesOutFilenameLow = "times"
timesOutFilenameHigh = ".out"
traceOutFilename = "trace"+today+".out"



#************************
#Top and Rst variables:
#************************
TYPE = {}

TIME_i = 0

NATOM = 0
NTYPES = 0

AMBER_ATOM_TYPE = []
CHARGE = []
ATOMIC_NUMBER = []
MASS = []

ACOEF = []
BCOEF = []

sigma = []
epsilon = []

POSITION = []
VELOCITY = []

BOX = {}

#************************
#Default Input Values:
#************************
imin = 0
ntb = 0
ntp=0
ntt = 1
nstlim =  999
dt = 0.001
vlimit = 1
ntf = 1
ibelly = 0
ntx = 5
irest = 1
ntpr = 1
ntwx = 1
ntwe = 1
temp0 =100.0
tempi = 0
tautp= 2.0
cut=12.0






#***********************************************************************
#****************************** Functions ******************************
#***********************************************************************
def openLennardTable():
  global lennardTableFilename, TYPE

  print endl + "****************************************************"
  print "Parsing Lennard Table"
  print "****************************************************"

  if not isfile(output_path + lennardTableFilename):
    print "The file ", output_path + lennardTableFilename, " doesn't exist."
    exit()
  
  lennard = open(output_path + lennardTableFilename, 'r')
  for l in lennard:
    l = l.strip()
    if not l.split() or l.startswith('<end>'):
      #print l
      break
    if l.startswith('#') or l.startswith('<start>'):
      #print l
      continue
    f = l.split()
    TYPE[f[0]] = {'sigma': float(f[1]), 'epsilon': float(f[2]),\
     'charge': float(f[3])}
  
  lennard.close()

#***********************************************************************


def parseTop(_top):
  global NATOM, NTYPES, AMBER_ATOM_TYPE, CHARGE, \
  ATOMIC_NUMBER, MASS, ACOEF, BCOEF, particlesFilename

  print endl + "****************************************************"
  print "Parsing the top file"
  print "****************************************************"
  
  if not isfile(_top):
    print "The file ", _top, " doesn't exist."
    exit()
  
  topFile = open(_top, 'r')
  
  topFile_iter = iter(topFile)
  for t in topFile_iter:
    if t.startswith("%FLAG POINTERS"):
      f = topFile_iter.next()
      if f.startswith("%FORMAT(10I8)"):
	l = (topFile_iter.next()).split()
	NATOM = int(l[0])
	NTYPES = int(l[1])
	#Here we can get all necesary parameters
      else:
	print "error: The file", _top, " isnt a prmtop file"
      print "NATOM: ", NATOM
      print "NTYPES: ", NTYPES
    
    if t.startswith("%FLAG AMBER_ATOM_TYPE"):
      t = topFile_iter.next()
      if t.startswith("%FORMAT(20a4)"):
	t = topFile_iter.next()
	while not t.startswith("%"):
	  l = t.split()
	  for name in l:
	    AMBER_ATOM_TYPE.append(name)
	  t = topFile_iter.next()
      else:
	print "error: The file", _top, " isnt a prmtop file"
      print "AMBER_ATOM_TYPE: ", AMBER_ATOM_TYPE
    
    if t.startswith("%FLAG CHARGE"):
      print "CHARGE"
      t = topFile_iter.next()
      if t.startswith("%FORMAT(5E16.8)"):
	t = topFile_iter.next()
	while not t.startswith("%"):
	  l = t.split()
	  for ch in l:
	    CHARGE.append(float(ch))
	  t = topFile_iter.next()
      else:
	print "error: The file", _top, " isnt a prmtop file"
      print "CHARGE: ", CHARGE
      print "len(CHARGE): ",len(CHARGE)
    
    if t.startswith("%FLAG MASS"):
      t = topFile_iter.next()
      if t.startswith("%FORMAT(5E16.8)"):
	t = topFile_iter.next()
	while not t.startswith("%"):
	  l = t.split()
	  for m in l:
	    MASS.append(float(m))
	  t = topFile_iter.next()
      else:
	print "error: The file", _top, " isnt a prmtop file"
      print "MASS: ", MASS
    
    
    
    
    # %FLAG LENNARD_JONES_ACOEF
    # %FORMAT(5E16.8)  (CN1(i), i=1,NTYPES*(NTYPES+1)/2)
    # CN1  : Lennard Jones r**12 terms for all possible atom type interactions,
    #       indexed by ICO and IAC; for atom i and j where i < j, the index into
    #       this array is as follows (assuming the value of ICO(index) is positive):
    #       CN1(ICO(NTYPES*(IAC(i)-1)+IAC(j))).
    
    if t.startswith("%FLAG LENNARD_JONES_ACOEF"):
      t = topFile_iter.next()
      if t.startswith("%FORMAT(5E16.8)"):
	t = (topFile_iter.next()).split()
	for acoef in t:
	  ACOEF.append(float(acoef))
      else:
	print "error: The file", _top, " isnt a prmtop file"
      print "ACOEF: ", ACOEF
      continue
    
    
    if t.startswith("%FLAG LENNARD_JONES_BCOEF"):
      t = topFile_iter.next()
      if t.startswith("%FORMAT(5E16.8)"):
	t = (topFile_iter.next()).split()
	for bcoef in t:
	  BCOEF.append(float(bcoef))
      else:
	print "error: The file", _top, " isnt a prmtop file"
      print "BCOEF: ", BCOEF
    
  
  #for debug
  for i in range(0, NTYPES*(NTYPES+1)/2):
    A = ACOEF[i]
    B = BCOEF[i]
    #sigma.append((A/B)**(1.0/6))
    sigma.append(0.0)
    epsilon.append(0.0)
    #epsilon.append((B**2)/(4*A))
  print "sigma = " + str(sigma)
  print "epsilon = " + str(epsilon)
  
  topFile.close()

#***********************************************************************


def parseRst(_rst):
  global NATOM, NTYPES, AMBER_ATOM_TYPE, CHARGE, POSITION, \
  ATOMIC_NUMBER, MASS, ACOEF, BCOEF, particlesFilename, \
  VELOCITY, TIME_i, BOX, ntb

  print endl + "****************************************************"
  print "Parsing the rst file"
  print "****************************************************"
  
  if not isfile(_rst):
    print "The file ", _rst, " doesn't exist."
    exit()
  
  rstFile = open(_rst, 'r')
  
  l = rstFile.readline()
  l = rstFile.readline().split()
  print l
  if int(l[0]) != NATOM:
    print "ERROR: the file ", _rst, " doesn\'t match with the top file"
    exit()
  if len(l) > 1:
    TIME_i = float(l[1])
  
  #POSITIONS
  for i in range(0, NATOM/2):
    f = rstFile.readline().split()
    POSITION.append({'x': float(f[0]), 'y': float(f[1]), 'z': float(f[2])})
    POSITION.append({'x': float(f[3]), 'y': float(f[4]), 'z': float(f[5])})
  if NATOM/2 < NATOM/2.0:
    f = rstFile.readline.split()
    POSITION.append({'x': float(f[0]), 'y': float(f[1]), 'z': float(f[2])})
  
  #VELOCITIES
  if rstFile.tell() == os.fstat(rstFile.fileno()).st_size:
    print "There are not velocities in the rst file"
    print "There aren\'t Box parameters"
    exit()
  
  f = rstFile.readline().split()
  #if there is box
  if ntb != 0:
    periodicity = True
    if rstFile.tell() == os.fstat(rstFile.fileno()).st_size:
      print "There are not velocities in the rst file"
      #f --> box
      #FORMAT(6F12.7) BOX(1), BOX(2), BOX(3)
      #BOX    : size of the periodic box
      print f
      BOX['x'] = float(f[0])
      BOX['y'] = float(f[1])
      BOX['z'] = float(f[2])
      BOX['alpha'] = float(f[3])
      BOX['beta'] = float(f[4])
      BOX['gamma'] = float(f[5])
      print "POS:", POSITION
      print "VEL:", VELOCITY
      print "BOX:", BOX
      rstFile.close()
      return
    else:
      print "levantar velocidades y despues caja"
  else:
    print "#levantar velocidades"
  
  for i in range(0, NATOM/2):
    VELOCITY.append({'x': float(f[0]), 'y': float(f[1]), 'z': float(f[2])})
    VELOCITY.append({'x': float(f[3]), 'y': float(f[4]), 'z': float(f[5])})
    if rstFile.tell() != os.fstat(rstFile.fileno()).st_size:
      f = rstFile.readline().split()
    else:
      rstFile.close()
      if(ntb != 0):
	print "ERROR There aren\'t Box parameters"
	exit()
      return
      
  if NATOM/2 < NATOM/2.0:
    VELOCITY.append({'x': float(f[0]), 'y': float(f[1]), 'z': float(f[2])})
    if rstFile.tell() != os.fstat(rstFile.fileno()).st_size:
      f = rstFile.readline().split()
    else:
      rstFile.close()
      if(ntb != 0):
	print "ERROR There aren\'t Box parameters"
	exit()
      return
    
  if f == []:
    rstFile.close()
    if(ntb != 0):
      print "ERROR There aren\'t Box parameters"
      exit()
    return
  #BOX
  #FORMAT(6F12.7) BOX(1), BOX(2), BOX(3)
  #BOX    : size of the periodic box
  BOX['x'] = float(f[0])
  BOX['y'] = float(f[1])
  BOX['z'] = float(f[2])
  BOX['alpha'] = float(f[3])
  BOX['beta'] = float(f[4])
  BOX['gamma'] = float(f[5])
  
  print "POS:", POSITION
  print "VEL:", VELOCITY
  print "BOX:", BOX
  rstFile.close()

#***********************************************************************
def parseMdin(_mdin):
  global NATOM, NTYPES, AMBER_ATOM_TYPE, CHARGE, POSITION, \
  ATOMIC_NUMBER, MASS, ACOEF, BCOEF, particlesFilename, \
  VELOCITY, TIME_i, imin, ntb, ntp, ntt, nstlim, dt, \
  vlimit, ntf, ibelly, ntx, irest, ntpr, ntwx, ntwe, \
  temp0, tempi, tautp, cut

  #print endl + "****************************************************"
  #print "Parsing the mdin file"
  #print "****************************************************"
  
  if not isfile(_mdin):
    print "The file ", _mdin, " doesn't exist."
    exit()
  
  mdinFile = open(_mdin, 'r')
  
  for l in mdinFile:
    if '&end' in l:
      break;
    noSpaces = l.replace(' ', '')
    parms = noSpaces.split(',')
    for parm in parms:
      setting = parm.split('=')
      for i in range(0,len(setting)):
	if setting[i] == 'temp0':
	  temp0 = float(setting[i+1])
	if setting[i] == 'tempi':
	  tempi = float(setting[i+1])
	if setting[i] == 'dt':
	  dt = float(setting[i+1])
	if setting[i] == 'nstlim':
	  nstlim = int(setting[i+1])
	if setting[i] == 'cut':
	  cut = float(setting[i+1])
	if setting[i] == 'ntb':
	  ntb = float(setting[i+1])
  
  print "dt = ", dt
  print "temp0 = ", temp0
  print "tautp = ", tautp
  print "nstlim = ", nstlim
  print "ntb = ", ntb
  print "cut = ", cut
  mdinFile.close()
    

#***********************************************************************



def setCutMdin(cutOff,_mdin, _mdin_modified):
  #print endl + "****************************************************"
  #print "Modifying the mdin file to " + _mdin_modified
  #print "****************************************************"

  if not isfile(_mdin):
    print "The file ", _mdin, " doesn't exist."
    exit()

  mdinModFile = open(_mdin_modified, 'w')
  mdinFile = open(_mdin, 'r')

  newCUT = 'cut = '+str(cutOff)+',\n'
  for l in mdinFile:
    if 'cut' in l:
      noSpaces = l.replace(' ', '')
      parms = noSpaces.split(',')
      ant = "  "
      for parm in parms:
        if 'cut' in parm:
          ant += newCUT
        elif not '\n' in parm:
          ant += parm + ','
      mdinModFile.write(ant)
    else:
      mdinModFile.write(l)
  mdinModFile.close()




def setStepsMdin(n,_mdin, _mdin_modified):
  #print endl + "****************************************************"
  #print "Modifying the mdin file to " + _mdin_modified
  #print "****************************************************"
  
  if not isfile(_mdin):
    print "The file ", _mdin, " doesn't exist."
    exit()
  
  mdinModFile = open(_mdin_modified, 'w')
  mdinFile = open(_mdin, 'r')
  
  newSTLIM = 'nstlim = '+str(n)+',\n'
  for l in mdinFile:
    if 'nstlim' in l:
      noSpaces = l.replace(' ', '')
      parms = noSpaces.split(',')
      ant = "  "
      for parm in parms:
	if 'nstlim' in parm:
	  ant += newSTLIM
	elif not '\n' in parm:
	  ant += parm + ','
      mdinModFile.write(ant)
    else:
      mdinModFile.write(l)
  mdinModFile.close()

#***********************************************************************

def makeParticlesInputFile():
  global NATOM, NTYPES, AMBER_ATOM_TYPE, CHARGE, POSITION, VELOCITY,\
  ATOMIC_NUMBER, MASS, ACOEF, BCOEF, BOX, particlesFilename

  print endl + "****************************************************"
  print "Making the inputFile"
  print "****************************************************"

  particlesFile = open(output_path + particlesFilename, 'w')
  
  #type, position, velocitie, charge
  particlesFile.write(str(NATOM)+endl)
  if(len(VELOCITY) > 0):
    for i in range(0, NATOM):
      particlesFile.write(AMBER_ATOM_TYPE[i] + tab + 
      str(POSITION[i]['x']) + tab + str(POSITION[i]['y']) + tab + str(POSITION[i]['z']) + tab +
      str(VELOCITY[i]['x']) + tab + str(VELOCITY[i]['y']) + tab + str(VELOCITY[i]['z']) + tab +
      str(CHARGE[i]) + endl)
  else:
    for i in range(0, NATOM):
      particlesFile.write(AMBER_ATOM_TYPE[i] + tab + 
      str(POSITION[i]['x']) + tab + str(POSITION[i]['y']) + tab + str(POSITION[i]['z']) + tab +
      str(0) + tab + str(0) + tab + str(0) + tab +
      str(CHARGE[i]) + endl)
    
  #box
  if ntb == 1:
    particlesFile.write(str(1) + tab)
    particlesFile.write(str(BOX['x']) + tab + str(BOX['y']) + tab + str(BOX['z']) + tab +
		        str(BOX['alpha']) + tab + str(BOX['beta']) + tab + str(BOX['gamma']) + endl )
  else:
    particlesFile.write(str(0) + endl)
    
  #parameters for run
  particlesFile.write(str(nstlim) + endl +
		      str(dt) + endl +
		      str(temp0) + endl +
		      str(tempi) + endl +
		      str(tautp) + endl +
		      str(cut) + endl)
  
  particlesFile.write( endl + "#Format:" + endl +
			"NATOM" + endl +
			"TYPE" + tab + "POS(x)" + tab + "POS(y)" + tab + "POS(z)" +
			tab + "VEL(x)" + tab + "VEL(y)" + tab + "VEL(z)"+
			tab + "CHARGE" + endl +
			"BOX?" + tab + "BOX(x)" + tab + "BOX(y)" + tab + "BOX(z)" + 
                        tab + "BOX(alpha)" + tab +  "BOX(beta)" + tab +  "BOX(gamma)" + endl + 
                        "NSTLIM" + endl +
			"dt" + endl +
			"temp0" + endl +
			"tempi" + endl +
			"tautp" + endl + 
			"cut" + endl)
  particlesFile.close()



#***********************************************************************

def makeLennardTable():
  global NATOM, NTYPES, AMBER_ATOM_TYPE, CHARGE, TYPE, \
  ATOMIC_NUMBER, MASS, ACOEF, BCOEF, lennardOutFilename, \
  lennardInputFilename

  print endl + "****************************************************"
  print "Making the lennardTable"
  print "****************************************************"

  lennard = open(output_path + lennardOutFilename, 'w')
  
  used_types = set(AMBER_ATOM_TYPE)
  
  lennard.write(str(NTYPES) + endl)
  for t in used_types:
    lennard.write(t + tab + 
		  str(TYPE[t]['sigma']) + tab + 
		  str(TYPE[t]['epsilon']) + tab + 
		  str(TYPE[t]['charge']) + endl)
  lennard.close()

#***********************************************************************

def runMacheAmber():
  global make, time, derivativeMode, analyticMode, ntb, coordenates, \
         output_Mache_Times_path, output_sander_Times_path, \
         timesOutFilenameLow, timesOutFilenameHigh, debug
  
  if make:
    makeCommand = "make"
    print makeCommand
    os.system(makeCommand)
  runCommand = "./amberMache "
  
  
  if ntb==1:
    runCommand += " -p "
  if cpu:
    runCommand += " -cpu " 
  if texture:
    runCommand += " -tex "
  if analyticMode:
    runCommand += " -a "
  if derivativeMode:
    runCommand += " -d "
  if coordenates:
    runCommand += " -c "
  #if textureSize:
   # runCommand += "-tex "+ str(texSize)
  if time:
    runCommand += " -t " + output_Mache_Times_path + timesOutFilenameLow + timesOutFilenameHigh
  else:
    if debug:
      runCommand += " -r "
    else:
      runCommand += " -ar "
  print runCommand
  os.system(runCommand)
  
  make = False

#***********************************************************************

def runSander(_top, _rst, _mdin, gpu):
  global time, nstlim, output_sander_Times_path, \
         output_sander_path, \
         timesOutFilenameLow, timesOutFilenameHigh
  
  if not os.path.exists(output_sander_path):
    os.makedirs(output_sander_path)
  else:
    for file in os.listdir(output_sander_path):
      if file.endswith("mdcrd") or file.endswith("mdinfo") or \
         file.endswith("mden") or file.endswith("restrt") or \
         file.endswith(outSanderFilename):
              os.system("rm " + output_sander_path + "/" + file)
  if gpu:
      runCommand = "pmemd.cuda_SPFP -O"
  else:
      runCommand = "sander -O"
      
  if time:
    if gpu:
      timesSanderFilename = output_sander_Times_path + timesOutFilenameLow + "_gpu_" + timesOutFilenameHigh
    else:
      timesSanderFilename = output_sander_Times_path + timesOutFilenameLow + timesOutFilenameHigh
    if not os.path.exists(output_sander_Times_path):
	os.makedirs(output_sander_Times_path)	

  runCommand += " -i " + _mdin
  runCommand += " -o " + outSanderFilename
  runCommand += " -p " + _top 
  runCommand += " -c " + _rst
  
  print runCommand
  os.system(runCommand)
  
  if gpu:
    output_sander_path_mov = output_sander_path + "Parallel/"
  else:
    output_sander_path_mov = output_sander_path + "Secuential/"
  
  if not os.path.exists(output_sander_path_mov):
	os.makedirs(output_sander_path_mov)
  
  moveCommand = "mv "
  for file in os.listdir("."):
    if file.endswith("mdcrd") or file.endswith("mdinfo") or \
      file.endswith("mden") or file.endswith("restrt") or \
      file.endswith(outSanderFilename):
	      os.system(moveCommand + file + " " + output_sander_path_mov)
  
  if time:
	print "Times in sander"
	with open(output_sander_path_mov + outSanderFilename, 'r') as outSander:
	    with open(timesSanderFilename, 'a+') as timesSander:
	      for l in outSander:
		  if "Elapsed(s)" in l:
		      l = l.replace('|', '').strip()
		      timesSander.write(str(nstlim)+ tab+ str(l.split()[2]))
		      timesSander.write(endl)

#***********************************************************************

def printHelp():
  print endl + "The usage mode is: "
  print "  python amber.py [options]"
  print endl + "   where options are:"
  print tab + "-P or -p :" + tab + "TopFile"
  print tab + "-C or -c :" + tab + "RstFile"
  print tab + "-I or -i :" + tab + "inputFile"
  print tab + "-T or -t :" + tab + "cantSteps   amountOfSamples   [-log]" 



  

#***********************************************************************
#**************************** Main Program *****************************
#***********************************************************************

top = input_amber_path + defaultTopFile
rst = input_amber_path + defaultRstFile
mdin = input_amber_path + defaultMdinFile


if len(sys.argv) < 2:
  print "\n Using all the default input files:"
  print tab + top
  print tab + rst
  print tab + mdin
  print "\n If you want a specific input file, see the help running:"
  print tab + "python amber.py -h"
else:
  
  for i in range(1,len(sys.argv)):
    arg = sys.argv[i]
    
    if (arg=='-H' or arg== '-h' or arg== '-help' or arg== '--help'):
      printHelp()
      exit()
    elif (arg=='-T' or arg== '-t'):
      steps = int(sys.argv[i+1])
      samples = int(sys.argv[i+2])
      time = True
      if (sys.argv[i+3]=='-log'):
	log = True
      else:
	log = False
    elif (arg=='-P' or arg== '-p') and (i < len(sys.argv)):
      top = sys.argv[i+1]
    elif (arg=='-C' or arg== '-c') and (i < len(sys.argv)):
      rst = sys.argv[i+1]
    elif (arg=='-I' or arg== '-i') and (i < len(sys.argv)):
      mdin = sys.argv[i+1]
    elif (arg=='-D' or arg== '-d'):
      debug = True
    elif (arg=='-deriv'):
	analyticMode=False
	derivativeMode=True
    elif(arg=='-tabla'):
	analyticMode=False
        derivativeMode=False
    elif(arg=='-cpu'):
	cpu=True
    elif(arg=='-tex'): 
	texture=True
	
  print "\n Using the input files:"
  print tab + top
  print tab + rst
  print tab + mdin

openLennardTable()

parseTop(top)
parseMdin(mdin)
parseRst(rst)
makeLennardTable()

if not time:
  analyticMode = False
  derivativeMode = False
  coordenates = True
  time = False
  makeParticlesInputFile()
  runMacheAmber()
  gpu = False
  runSander(top, rst, mdin, gpu)
  gpu = True
  runSander(top, rst, mdin, gpu)
  
else:
  print endl + "****************************************************"
  print "Running time by steps"
  print "****************************************************"
  print " Results in the file  " + output_Mache_path+traceOutFilename + endl
  
  directory = str(steps) +"_"+ str(samples)+"/"
  if not os.path.exists(output_Mache_path+directory):
    os.makedirs(output_Mache_path+directory)
  output_Mache_Times_path = output_Mache_path + directory
  output_sander_Times_path = output_sander_path + directory
  #****Redirect stdout to a file****
  #sys.stdout = open(output_Mache_path+directory+traceOutFilename, 'w')
  
  #i = 1
  #var = steps/samples
  # if log:
  #   i = 0
  #   maximo = steps
  #   while maximo > 1:
  #     maximo /= 10
  #     i+=1
 
  for cutoff in range(10,60,4):
      n=steps 
      #for n in range(0,steps,samples):
      mdin_modified = mdin + "_" + str(n)
      setStepsMdin(n, mdin, mdin_modified)
      parseMdin(mdin_modified)
      cut=cutoff
      makeParticlesInputFile()
     
      debug=True
      timesOutFilenameHigh = ".out"
      
      texture=False
      cpu=False
      analyticMode=False
      derivativeMode=False
      


      #analitico sobre cpu
      cpu=True
      analyticMode=True  
      #runMacheAmber()
    
      #tabla sobre cpu
      analyticMode=False
      #runMacheAmber()
 
      #derivada sobre cpu
      derivativeMode=True
      #runMacheAmber()
      
      #analitico sobre gpu
      cpu=False
      analyticMode=True 
      derivativeMode=False   
      runMacheAmber()
      
      #tabla SIN textura gpu
      analyticMode=False
      runMacheAmber()

      #tabla CON textura gpu
      texture=True
      runMacheAmber()
 
      #derivada SIN textura gpu
      derivativeMode=True
      texture=False
      runMacheAmber()

      #derivada CON textura gpu      
      texture=True
      runMacheAmber()



      
      os.remove(mdin_modified)
      print "--------------------------------------------------------------------"
    
