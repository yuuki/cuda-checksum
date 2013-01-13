#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <timer.h>


#include "cksum.h"

#define MAX_BUFSIZE 1500

static void print_result(uint16_t cpu_cksum, uint16_t gpu_cksum, float latency) {
    printf("-------------------------\n");
    /* printf("cuda kernel elapsed time %4.2f\n", kernel_elapsed_time); */
    printf("latency time %8.2f ms\n", latency);
    printf("CPU checksum %2x\n", cpu_cksum);
    printf("GPU checksum %2x\n", gpu_cksum);
    (cpu_cksum == gpu_cksum) ? printf("Test passed\n") : printf("test failed\n");
    printf("-------------------------\n");
}

static inline uint16_t cpu_cksum(uint8_t* data, size_t len) {
    uint32_t sum = 0, c;
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

int main(int argc, char* argv[]) {
    int ret = 0;
    extern char *optarg;
    extern int  optind, opterr;
    char ch;

    int num_tblocks = 1, num_threads = 1;
    int devID = 0;
    cudaDeviceProp deviceProp;

    devID = findCudaDevice(argc, (const char **)argv);

    while ((ch = getopt(argc, argv, "b:c:n:t:")) != -1) {
        switch(ch) {
            case 'b':
                num_tblocks = atoi(optarg);
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
        }
    }
    argc -= optind;
    argv += optind;

    if (argc > 1) {
        printf("too many operands\n");
        return -1;
    } else {
        // This will pick the best possible CUDA capable device

        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

        if (deviceProp.major < 3) {
            printf("cudaChecksum requires a GPU with compute capability "
                    "3.0 or later, exiting...\n");
            cudaDeviceReset();
            exit(EXIT_SUCCESS);
        }

        char *buf = (char *)malloc(sizeof(char) * MAX_BUFSIZE);
        memset(buf, 0, sizeof(char) * MAX_BUFSIZE);
        if (fgets((char *)buf, sizeof(char) * MAX_BUFSIZE, stdin) == NULL) {
            perror("fgets");
            checkCudaErrors(cudaFreeHost(buf));
            return -1;
        }

        StartTimer();

        size_t buflen = strlen((char *)buf);

        uint16_t gpu_cksum = 0;
        int ret = cu_cksum(&gpu_cksum, (uint32_t *)buf, buflen, num_tblocks, num_threads);
        if (ret < 0) {
            checkCudaErrors(cudaFreeHost(buf));
            return -1;
        }

        float latency = GetTimer();

        uint16_t _cpu_cksum = cpu_cksum((uint8_t *)buf, buflen);
        print_result(_cpu_cksum, gpu_cksum, latency);

        free(buf);
        cudaDeviceReset();
    }

    return 0;
}
