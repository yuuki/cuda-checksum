BIN               := cucksum

CUDA_INSTALL_PATH := /usr/local/cuda
CUDA_SDK_PATH     := $(CUDA_INSTALL_PATH)/NVIDIA_GPU_Computing_SDK/C
INCLUDES          := -I$(CUDA_SDK_PATH)/common/inc
LIBS              := -L$(CUDA_SDK_PATH)/lib -L$(CUDA_INSTALL_PATH)/lib64 -lcutil_x86_64 -lcudart  -lstdc++
CFLAGS            := -O2 -Wall -Wextra
NVCCFLAGS         := -G2 -arch=sm_20
LDFLAGS           :=

NVCC              := $(CUDA_INSTALL_PATH)/bin/nvcc
CC                := gcc
LINKER            := gcc

C_SOURCES         := $(wildcard *.c)
CU_SOURCES        := $(wildcard *.cu)
HEADERS           := $(wildcard *.h)
C_OBJS            := $(patsubst %.c, %.o, $(C_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.o, $(CU_SOURCES))

$(BIN): clean $(C_OBJS) $(CU_OBJS) $(HEADERS)
	$(LINKER) -o $(BIN) $(CU_OBJS) $(C_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)

$(C_OBJS): $(C_SOURCES) $(HEADERS)
	$(CC) -c $(C_SOURCES) $(CFLAGS) $(INCLUDES)

$(CU_OBJS): $(CU_SOURCES) $(HEADERS)
	$(NVCC) -c $(CU_SOURCES) $(INCLUDES) $(NVCCFLAGS)

run: $(BIN)
	LD_LIBRARY_PATH=$(CUDA_INSTALL_PATH)/lib ./$(BIN)

clean:
	rm -f $(BIN) *.o
