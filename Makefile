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
#NVCC  ?= $(CUDA_PATH)/bin/nvcc $(GCC)

################################################################################

# Dependences

#CFILES ?= main.cu

################################################################################


#CC := g++ # This is the main compiler

SRCDIR := src
BUILDDIR := build
BINDIR := bin
TARGET := $(BINDIR)/pastorius
 
#SRCEXT := cu
#SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
#OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
#CFLAGS := -g # -Wall
#LIB := -pthread -lmongoclient -L lib -lboost_thread-mt -lboost_filesystem-mt -lboost_system-mt
#INC := -I include

SOURCES := $(SRCDIR)/main.cu
OBJECTS := $(BUILDDIR)/main.o


$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@echo " $(NVCC) $^ -o $(TARGET) $(LIB)"; $(CC) $^ -o $(TARGET) $(LIB)
  	
$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR) 
	@mkdir -p $(BINDIR)
	#@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<
	$(NVCC) $(GENCODE_SM20) $(ALL_LDFLAGS) -DTIME $(CUDART) -o $@ -c $<

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)

# Tests
tester:
	$(CC) $(CFLAGS) test/tester.cpp $(INC) $(LIB) -o bin/tester

# Spikes
ticket:
	$(CC) $(CFLAGS) spikes/ticket.cpp $(INC) $(LIB) -o bin/ticket
