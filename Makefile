BIN               := checksum_test
CUDA_INSTALL_PATH := /usr/local/cuda

NVCC              := $(CUDA_INSTALL_PATH)/bin/nvcc
CC                := g++
LINKER            := g++
AR                := ar

INCLUDES          := -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc -Iinclude
LIBS              := -lcudart  -lstdc++
CFLAGS            := -O2 -Wall -Wextra
NVCCFLAGS         := -O2 -gencode=arch=compute_30,code=sm_30
LDFLAGS           := -L$(CUDA_INSTALL_PATH)/lib64
ARFLAGS           := crsv

SRC_DIR           := src
TEST_DIR 				  := test
TEST_OBJS      		:= $(TEST_DIR)/cksum_test.o
CU_OBJS        		:= $(SRC_DIR)/cksum.o
AR_OBJ            := libcudacksum.a

all: $(BIN)

$(BIN): clean $(TEST_OBJS) $(CU_OBJS)
	$(LINKER) -o $(BIN) $(TEST_OBJS) $(CU_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)
	$(AR) $(ARFLAGS) $(AR_OBJ) $(CU_OBJS)

test: $(BIN)
	LD_LIBRARY_PATH=$(CUDA_INSTALL_PATH)/lib64 ./$(BIN)

.SUFFIXES: .o .c .cpp .cu

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

.cu.o:
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

run: $(BIN)
	LD_LIBRARY_PATH=$(CUDA_INSTALL_PATH)/lib64 ./$(BIN)

clean:
	rm -f $(BIN) $(TEST_OBJS) $(CU_OBJS)
