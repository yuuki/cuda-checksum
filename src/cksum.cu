#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

extern __shared__ uint32_t shared_mem[];

__device__ uint16_t d_cksum;


__device__ inline void device_memset(uint8_t* buf, int x, size_t n) {
    for (int i = 0; i < n; i++) {
        buf[i] = (uint8_t)x;
    }
}

__device__ inline uint8_t* device_memcpy(uint8_t* dst,  const uint8_t* src, size_t n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
    return dst;
}

__device__ inline uint16_t checksum(const uint8_t* data, const size_t len) {
    uint32_t sum = 0;
    size_t c = 0;
    uint16_t *ptr;
    ptr = (uint16_t *)data;
    for (c = len; c > 1; c-= 2) {
        sum += *ptr;
        if (sum & 0x80000000) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        ++ptr;
    }
    if (c == 1) {
        uint16_t val = 0;
        memcpy(&val, ptr, sizeof(uint8_t));
        sum += val;
    }
    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    return (sum == 0xFFFF) ? sum : ~sum;
}

__global__ void kernelChecksum(const uint32_t* g_buf, const size_t buflen) {
    /*const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;*/
    uint32_t *data = shared_mem;

    memcpy((uint8_t *)data, (uint8_t *)g_buf, buflen);
    d_cksum = checksum((uint8_t *)data, buflen);

    __syncthreads();
}

uint16_t cu_cksum(uint16_t *cksum, const uint32_t *buf, const size_t buflen, const int num_threads, const int num_tblocks) {
    const size_t mem_size = sizeof(uint32_t) * buflen * num_threads * num_tblocks;

    uint32_t *d_buf   = NULL;
    checkCudaErrors(cudaMalloc((void** )&d_buf, mem_size));
    checkCudaErrors(cudaMemcpy(d_buf, buf, mem_size, cudaMemcpyHostToDevice));

    /*cudaEvent_t start, stop;*/
    /*checkCudaErrors(cudaEventCreate(&start));*/
    /*checkCudaErrors(cudaEventCreate(&stop));*/

    /*checkCudaErrors(cudaEventRecord(start, 0));*/

    kernelChecksum <<<num_tblocks, num_threads, buflen>>> (d_buf, buflen);
    getLastCudaError("Kernel Execution failed");

    checkCudaErrors(cudaDeviceSynchronize());

    /*checkCudaErrors(cudaEventRecord(stop, 0));*/
    /*checkCudaErrors(cudaEventSynchronize(stop));*/

    /*float kernel_time = 0.0f;*/
    /*checkCudaErrors(cudaEventElapsedTime(&kernel_time, start, stop));*/


    typeof(d_cksum) _cksum;
    checkCudaErrors(cudaMemcpyFromSymbol(&_cksum, d_cksum, sizeof(_cksum), 0, cudaMemcpyDeviceToHost));


    checkCudaErrors(cudaFree(d_buf));
    /*checkCudaErrors(cudaEventDestroy(start));*/
    /*checkCudaErrors(cudaEventDestroy(stop));*/

    cudaDeviceReset();

    *cksum = _cksum;

    return 0;
}
