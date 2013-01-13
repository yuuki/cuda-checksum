BIN               := checksum_test

CUDA_INSTALL_PATH := /usr/local/cuda
INCLUDES          := -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc -Iinclude
LIBS              := -lcudart -lstdc++
CFLAGS            := -O0 -g -Wall -Wextra
NVCCFLAGS         := -O0 -g -G -gencode=arch=compute_30,code=sm_30
LDFLAGS           := -L$(CUDA_INSTALL_PATH)/lib64

NVCC              := $(CUDA_INSTALL_PATH)/bin/nvcc
CC                := g++
LINKER            := g++
SRC_DIR           := src
TEST_DIR 				  := test
TEST_OBJS      		:= $(TEST_DIR)/cksum_test.o
CU_OBJS        		:= $(SRC_DIR)/cksum.o

all: $(BIN)

$(BIN): clean $(TEST_OBJS) $(CU_OBJS)
	$(LINKER) -o $(BIN) $(TEST_OBJS) $(CU_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)

.SUFFIXES: .o .c .cpp .cu

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

.cu.o:
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(BIN) $(TEST_OBJS) $(CU_OBJS)
