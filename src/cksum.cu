
#include <stdint.h>
#include <stdio.h>

#include "cksum.cuh"


__device__ inline void device_memset(uint8* buf, int x, size_t n) {
    for (int i = 0; i < n; i++) {
        buf[i] = (uint8)x;
    }
}

__device__ inline uint8* device_memcpy(uint8* dst,  uint8* src, size_t n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
    return dst;
}

__device__ inline uint16_t checksum(uint8_t* data, size_t len) {
    uint32_t sum = 0, c;
    uint16_t val, *ptr;
    ptr = (uint16_t *data);
    for (c = len; c > 1; c-= 2) {
        sum += *ptr;
        if (sum & 0x80000000) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        ++ptr;
    }
    if (c == 1) {
        val = 0;
        device_memcpy(&val, ptr, sizeof(uint8_t);
        sum += val;
    }
    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    return (sum == 0xFFFF) ? sum : ~sum;
}

__global__ void kernel_cksum(uint8_t* g_buf, uint32_t* g_buflen, uint16_t* g_cksum) {
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ uint32_t data[*g_buflen];

    device_memcpy(data, g_buf, sizeof(data));

    *g_cksum = checksum(data, sizeof(data));

    _syncthreads();
}
