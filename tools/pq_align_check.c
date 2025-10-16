#include <stdio.h>
#include <stddef.h>
#include "pq_encode.h"

int main(void) {
    printf("C sizeof(PQLayout): %zu\n", sizeof(PQLayout));
    printf("C alignof(PQLayout): %zu\n", _Alignof(PQLayout));
    printf("C sizeof(bool): %zu\n", sizeof(bool));
    printf("C sizeof(int): %zu\n", sizeof(int));

    printf("C sizeof(PQEncodeOpts): %zu\n", sizeof(PQEncodeOpts));
    printf("C alignof(PQEncodeOpts): %zu\n", _Alignof(PQEncodeOpts));
    printf("C offsetof(PQEncodeOpts, layout): %zu\n", offsetof(PQEncodeOpts, layout));
    printf("C offsetof(PQEncodeOpts, use_dot_trick): %zu\n", offsetof(PQEncodeOpts, use_dot_trick));
    printf("C offsetof(PQEncodeOpts, precompute_x_norm2): %zu\n", offsetof(PQEncodeOpts, precompute_x_norm2));
    printf("C offsetof(PQEncodeOpts, prefetch_distance): %zu\n", offsetof(PQEncodeOpts, prefetch_distance));
    printf("C offsetof(PQEncodeOpts, num_threads): %zu\n", offsetof(PQEncodeOpts, num_threads));
    printf("C offsetof(PQEncodeOpts, soa_block_B): %zu\n", offsetof(PQEncodeOpts, soa_block_B));
    printf("C offsetof(PQEncodeOpts, interleave_g): %zu\n", offsetof(PQEncodeOpts, interleave_g));

    /* Expected on LP64 (macOS/iOS):
       - sizeof(PQLayout) = 4 (C int)
       - sizeof(bool) = 1
       - sizeof(int) = 4
       - sizeof(PQEncodeOpts) = 24
       - alignof(PQEncodeOpts) = 4
       Offsets: layout=0, use_dot=4, precompute=5, prefetch=8, num_threads=12, soa_block_B=16, interleave_g=20
    */

    return 0;
}

