#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>

#include "cksum.h"

extern "C"

static int loop_cksum(uint32_t *data, size_t len, int max_exe_times, int num_threads, int num_tblocks) {
    uint32_t* buf;
    unsigned int mem_size = sizeof(uint8_t) * block_size * num_threads * num_tblocks;
    unsigned int bufsize = sizeof(int) * num_threads * num_tblocks;
    unsigned int md_size = sizeof(int) * SHA_DIGEST_LENGTH * num_threads * num_tblocks;

    cutilSafeCall(cudaHostAlloc((void**)&buf, mem_size * max_exe_times, 0));
    cutilSafeCall(cudaHostAlloc((void**)&msglen, msglen_size * max_exe_times, 0));
    cutilSafeCall(cudaHostAlloc((void**)&d_cksum, md_size * max_exe_times, 0));

    uint32_t bandwidth_timer = 0;
    cutilCheckError(cutCreateTimer(&bandwidth_timer));
    cutilCheckError(cutStartTimer(bandwidth_timer));

    uint16_t d_cksum = 0;
    cutilSafeCall(cudaMalloc((void* )&d_cksum, sizeof(uint16_t)));

    cudaEvent_t start, stop;
    cutilSafeCall(cudaEventCreate(&start));
    cutilSafeCall(cudaEventCreate(&stop));

    cutilSafeCall(cudaEventRecord(start, 0));

    for (int i = 0; i < exe_times; i++) {
        unsigned int offset = num_tblocks * num_threads * i;
        do_sha1<<<num_tblocks, num_threads>>>(d_buf + block_size * offset , d_msglen + offset, d_md + SHA_DIGEST_LENGTH * offset);
        cutilCheckMsg("Kernel execution failed");
    }

    cutilSafeCall(cudaEventRecord(stop, 0));
    cutilSafeCall(cudaEventSynchronize(stop));
    float kernel_time = 0.0f;
    cutilSafeCall(cudaEventElapsedTime(&kernel_time, start, stop));

    cutilSafeCall(cudaMemcpyAsync(md, d_md, md_size, cudaMemcpyDeviceToHost, 0));

    cutilSafeCall(cutilDeviceSynchronize());

    cutilCheckError(cutStopTimer(bandwidth_timer));
    double bandwidth_time = cutGetTimerValue(bandwidth_timer);
    cutilCheckError(cutDeleteTimer(bandwidth_timer));

    cutilSafeCall(cudaFreeHost(buf));
    cutilSafeCall(cudaFreeHost(msglen));
    cutilSafeCall(cudaFreeHost(md));
    cutilSafeCall(cudaFree(d_cksum));
    cutilSafeCall(cudaEventDestroy(start));
    cutilSafeCall(cudaEventDestroy(stop));

    cutilDeviceReset();
}

int main(int argc, char* argv[]) {
    extern char *optarg;
    extern int  optind, opterr;

    cutilSafeCall(cudaSetDevice(cutGetMaxGflopsDeviceId()));

    while ((ch = getopt(argc, argv, "b:c:n:t:")) != -1) {
        switch(ch) {
            case 'b':
                num_tblocks = atoi(optarg);
                break;
            case 'c':
                block_size = atoi(optarg);
                break;
            case 'n':
                max_exe_times = atoi(optarg);
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
        }
    }
    argc -= optind;
    argv += optind;

    if (argc == 0) {
        printf("no operand\n");
        return -1;
    } else if (argc > 1) {
        printf("too many operands\n");
        return -1;
    } else {
        IN = fopen(*argv,"rb");
        if (IN == NULL) {
            perror(*argv);
            err++;
        }
        do_fp(IN);
        fclose(IN);
    }
}
