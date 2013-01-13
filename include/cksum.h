#ifndef _CKSUM_H_
#define _CKSUM_H_

#include <stdint.h>

extern uint16_t cu_cksum(uint16_t *cksum, const uint32_t *buf, const size_t buflen, const int num_threads, const int num_tblocks);

#endif // _CKSUM_H_
