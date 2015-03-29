# Location of the CUDA Toolkit
CUDA_PATH ?= "/usr/local/cuda"

# internal flags
NVCCFLAGS   := -m64

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(NVCCLDFLAGS)

# Common includes and paths for CUDA
#-I../../common/inc
INCLUDES  := -I../inc

################################################################################

# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)
CUDART		:= -lcudart

################################################################################

# Compiler flags

# GCC   ?= g++
NVCC  ?= $(CUDA_PATH)/bin/nvcc $(GCC)

################################################################################

# Dependences

CFILES ?= main.cu

################################################################################

# Target rules
all: build

build: main

main.o: $(CFILES)
	$(NVCC) $(GENCODE_SM20) $(ALL_LDFLAGS) -DTIME $(CUDART) -o $@ -c $<

main: main.o
	$(NVCC) $(GENCODE_SM20) $(ALL_LDFLAGS)  -o amberMache $+

# nvcc -gencode=arch=compute_20,code=sm_20 -m64 main.cu -lcudart

run: build
	./amberMache

clean:
	rm -f amberMache main.o *~

clobber: clean

# /usr/lib/nvidia-cuda-toolkit/bin/nvcc -ccbin g++ -I../inc  -m64    -gencode arch=compute_10,code=sm_10 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\" -o deviceQuery.o -c deviceQuery.cpp
# /usr/lib/nvidia-cuda-toolkit/bin/nvcc -ccbin g++   -m64        -o deviceQuery deviceQuery.o 

# # Target rules
# all: build
# 
# build: deviceQuery
# 
# deviceQuery.o: deviceQuery.cpp
# 	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
# 
# deviceQuery: deviceQuery.o
# 	$(NVCC) $(ALL_LDFLAGS) -o $@ $+
# 
# 
# run: build
# 	./deviceQuery
# 
# clean:
# 	rm -f deviceQuery deviceQuery.o
# 
# clobber: clean
