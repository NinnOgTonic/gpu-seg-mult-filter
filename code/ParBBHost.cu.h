#ifndef PAR_BB_HOST
#define PAR_BB_HOST

#include "ParBBKernels.cu.h"

#include <sys/time.h>
#include <time.h>

int nextMultOf(unsigned int x, unsigned int m) {
    if( x % m ) return x - (x % m) + m;
    else        return x;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

/**
 * block_size is the size of the cuda block (must be a multiple
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * d_out      is the output GPU array -- if you want
 *            its data on CPU needs to copy it back to host.
 *
 * OP         class denotes the associative binary operator
 *                and should have an implementation similar to
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates,
 *                e.g., float or int.
 */
template<class OP, class T>
void scanInc(    unsigned int  block_size,
                 unsigned long d_size,
                 T*            d_in,  // device
                 T*            d_out  // device
) {
    unsigned int num_blocks;
    unsigned int sh_mem_size = block_size * 32; //sizeof(T);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    scanIncKernel<OP,T><<< num_blocks, block_size, sh_mem_size >>>(d_in, d_out, d_size);
    cudaThreadSynchronize();

    if (block_size >= d_size) { return; }

    /**********************/
    /*** Recursive Case ***/
    /**********************/

    //   1. allocate new device input & output array of size num_blocks
    T *d_rec_in, *d_rec_out;
    cudaMalloc((void**)&d_rec_in , num_blocks*sizeof(T));
    cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T));

    unsigned int num_blocks_rec = ( (num_blocks % block_size) == 0 ) ?
                                  num_blocks / block_size     :
                                  num_blocks / block_size + 1 ;

    //   2. copy in the end-of-block results of the previous scan
    copyEndOfBlockKernel<T><<< num_blocks_rec, block_size >>>(d_out, d_rec_in, num_blocks);
    cudaThreadSynchronize();

    //   3. scan recursively the last elements of each CUDA block
    scanInc<OP,T>( block_size, num_blocks, d_rec_in, d_rec_out );

    //   4. distribute the the corresponding element of the
    //      recursively scanned data to all elements of the
    //      corresponding original block
    distributeEndBlock<OP,T><<< num_blocks, block_size >>>(d_rec_out, d_out, d_size);
    cudaThreadSynchronize();

    //   5. clean up
    cudaFree(d_rec_in );
    cudaFree(d_rec_out);
}


/**
 * block_size is the size of the cuda block (must be a multiple
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * flags      is the flag array, in which !=0 indicates
 *                start of a segment.
 * d_out      is the output GPU array -- if you want
 *            its data on CPU you need to copy it back to host.
 *
 * OP         class denotes the associative binary operator
 *                and should have an implementation similar to
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates,
 *                e.g., float or int.
 */
template<class OP, class T>
void sgmScanInc( const unsigned int  block_size,
                 const unsigned long d_size,
                 T*            d_in,  //device
                 int*          flags, //device
                 T*            d_out  //device
) {
    unsigned int num_blocks;
    //unsigned int val_sh_size = block_size * sizeof(T  );
    unsigned int flg_sh_size = block_size * sizeof(int);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    T     *d_rec_in;
    int   *f_rec_in;
    cudaMalloc((void**)&d_rec_in, num_blocks*sizeof(T  ));
    cudaMalloc((void**)&f_rec_in, num_blocks*sizeof(int));

    sgmScanIncKernel<OP,T> <<< num_blocks, block_size, 32*block_size >>>
                    (d_in, flags, d_out, f_rec_in, d_rec_in, d_size);
    cudaThreadSynchronize();
    //cudaError_t err = cudaThreadSynchronize();
    //if( err != cudaSuccess)
    //    printf("cudaThreadSynchronize error: %s\n", cudaGetErrorString(err));

    if (block_size >= d_size) { cudaFree(d_rec_in); cudaFree(f_rec_in); return; }

    //   1. allocate new device input & output array of size num_blocks
    T   *d_rec_out;
    int *f_inds;
    cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T   ));
    cudaMalloc((void**)&f_inds,    d_size    *sizeof(int ));

    //   2. recursive segmented scan on the last elements of each CUDA block
    sgmScanInc<OP,T>
                ( block_size, num_blocks, d_rec_in, f_rec_in, d_rec_out );

    //   3. create an index array that is non-zero for all elements
    //      that correspond to an open segment that crosses two blocks,
    //      and different than zero otherwise. This is implemented
    //      as a CUDA-block level inclusive scan on the flag array,
    //      i.e., the segment that start the block has zero-flags,
    //      which will be preserved by the inclusive scan.
    scanIncKernel<Add<int>,int> <<< num_blocks, block_size, flg_sh_size >>>
                ( flags, f_inds, d_size );

    //   4. finally, accumulate the recursive result of segmented scan
    //      to the elements from the first segment of each block (if
    //      segment is open).
    sgmDistributeEndBlock <OP,T> <<< num_blocks, block_size >>>
                ( d_rec_out, d_out, f_inds, d_size );
    cudaThreadSynchronize();

    //   5. clean up
    cudaFree(d_rec_in );
    cudaFree(d_rec_out);
    cudaFree(f_rec_in );
    cudaFree(f_inds   );
}

/**
 * d_in       is the device matrix; it is supposably
 *                allocated and holds valid values (input).
 *                semantically of size [height x width]
 * d_out      is the output GPU array -- if you want
 *            its data on CPU needs to copy it back to host.
 *                semantically of size [width x height]
 * height     is the height of the input matrix
 * width      is the width  of the input matrix
 */

template<class T, int tile>
void transposePad( T*                 inp_d,
                T*                 out_d,
                const unsigned int height,
                const unsigned int width,
                const unsigned int oinp_size,
                T                  pad_elem
) {
   // 1. setup block and grid parameters
   int  dimy = ceil( ((float)height)/tile );
   int  dimx = ceil( ((float) width)/tile );
   dim3 block(tile, tile, 1);
   dim3 grid (dimx, dimy, 1);

   //2. execute the kernel
   matTransposeTiledPadKer<T,tile> <<< grid, block >>>
    (inp_d, out_d, height, width, oinp_size,pad_elem);

   cudaThreadSynchronize();
}

void cuprintf(int *cudabuf, int size) {
    int *hostbuf;
    hostbuf = (int*)malloc(size * sizeof(int));
    cudaMemcpy(hostbuf, cudabuf, size * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n");
    for(int i = 0; i < size; i++) {
        printf("%d, ", hostbuf[i]);
    }
    printf("\n");
}
void cuprintff(float *cudabuf, int size) {
    float *hostbuf;
    hostbuf = (float*)malloc(size * sizeof(float));
    cudaMemcpy(hostbuf, cudabuf, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\n");
    for(int i = 0; i < size; i++) {
        printf("%f, ", hostbuf[i]);
    }
    printf("\n");
}
void cuprintf4(MyInt4 *cudabuf, int size) {
    MyInt4 *hostbuf;
    hostbuf = (MyInt4*)malloc(size * sizeof(int) * 4);
    cudaMemcpy(hostbuf, cudabuf, size * sizeof(int) * 4, cudaMemcpyDeviceToHost);
    printf("\n");
    for(int i = 0; i < size; i++) {
        printf("%3d:    %d, %d, %d, %d\n", i, hostbuf[i].x, hostbuf[i].y, hostbuf[i].z, hostbuf[i].w);
    }
    printf("\n");
}

#define MAX_BLOCKS 65535

/**
 * block_size is the size of the cuda block (must be a multiple
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * d_out      is the output GPU array -- if you want
 *            its data on CPU needs to copy it back to host.
 *
 * COND       class denotes the partitioning function (condition)
 *                and should have an implementation similar to
 *                `class LessThan' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions,
 *                and types `InType` and `OutType'
 */
template<class COND>
int filterTrad( const unsigned int     num_elems,
                const unsigned int     num_hwd_thds,
                typename COND::InType* d_in,  // device
                typename COND::InType* d_out  // device
) {
    struct timeval t_start, t_med1, t_med2, t_end, t_diff;
    unsigned long int elapsed;
    unsigned int block_size, num_blocks;
    unsigned int filt_size;
    int *cond_res, *inds_res;
    //block_size = 960;

    block_size = nextMultOf( (num_elems + MAX_BLOCKS - 1) / MAX_BLOCKS, 32 );
    block_size = (block_size < 256) ? 256 : block_size;
    num_blocks = (num_elems + block_size - 1) / block_size;

    // allocate the boolean & index arrays
    cudaMalloc((void**)&cond_res, num_elems*sizeof(int));
    cudaMalloc((void**)&inds_res, num_elems*sizeof(int));

    // map the condition
    gettimeofday(&t_start, NULL);
    mapKernel<COND><<<num_blocks, block_size>>>(d_in, cond_res, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med1, NULL);
    //cuprintff(d_in, 100);
    //cuprintf(cond_res, 100);

    // inclusive scan of the condition results
    scanInc<Add<int>,int>(block_size, num_elems, cond_res, inds_res);
    cudaThreadSynchronize();
    gettimeofday(&t_med2, NULL);
    //cuprintf(inds_res, 100);

    cudaMemcpy(&filt_size, &inds_res[num_elems-1], sizeof(int), cudaMemcpyDeviceToHost);
    writeKernel<typename COND::InType><<<num_blocks, block_size>>>(d_in, inds_res, d_out, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_end, NULL);
    //cuprintff(d_out, 100);

    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Traditional Filter total runtime: %lu microsecs, from which:\n", elapsed);

    timeval_subtract(&t_diff, &t_med1, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Map Cond Kernel runs in: %lu microsecs\n", elapsed);

    timeval_subtract(&t_diff, &t_med2, &t_med1);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Scan Addition Kernel runs in: %lu microsecs\n", elapsed);

    timeval_subtract(&t_diff, &t_end, &t_med2);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Global Write Kernel runs in: %lu microsecs\n", elapsed);

    // free resources
    cudaFree(inds_res);
    cudaFree(cond_res);

    return filt_size;
}

template<class COND>
int filterTradChunked(  const unsigned int     num_elems,
                        const unsigned int     num_hwd_thds,
                        typename COND::InType* d_in,  // device
                        typename COND::InType* d_out,  // device
                        const int VERSION

) {
    // compute a suitable CHUNK factor and padd the intermediate arrays such
    // that 64 | D_HEIGHT and 32 | D_WIDTH
    const unsigned int D_WIDTH = min( nextMultOf(max(num_elems/num_hwd_thds,1), 32), 384);  // SEQ CHUNK
    const unsigned int D_HEIGHT= nextMultOf( (num_elems + D_WIDTH - 1) / D_WIDTH, 64 );
    const unsigned int PADD    = nextMultOf(D_HEIGHT*D_WIDTH, 64*D_WIDTH) - num_elems;

    struct timeval t_start, t_med0, t_med1, t_med2, t_end, t_diff;
    unsigned long int elapsed;

    typename COND::InType *d_tr_in;
    int *cond_res, *inds_res;
    int filt_size;
    cudaMalloc((void**)&d_tr_in, D_HEIGHT*D_WIDTH*sizeof(typename COND::InType));
    cudaMalloc((void**)&cond_res, D_HEIGHT*D_WIDTH*sizeof(int));
    cudaMalloc((void**)&inds_res, 2*D_HEIGHT*sizeof(int));

    gettimeofday(&t_start, NULL);
    { // 1. Transpose with padding!
       transposePad<typename COND::InType,16>
            (d_in, d_tr_in, D_HEIGHT, D_WIDTH, num_elems, COND::padelm);
    }
    cudaThreadSynchronize();
    gettimeofday(&t_med0, NULL);
    //cuprintff(d_in, 100);
    //cuprintff(d_tr_in, 100);

    { // 2. The Map Condition Kernel Call
        const unsigned int block_size = 64; //256;
        const unsigned int num_blocks = (D_HEIGHT + block_size - 1) / block_size;
        // map the condition
        mapChunkKernel<COND><<<num_blocks, block_size>>>
                ( d_tr_in, cond_res, inds_res, D_HEIGHT, D_WIDTH );
    }
    cudaThreadSynchronize();
    gettimeofday(&t_med1, NULL);
    //cuprintf(cond_res, 100);
    //cuprintf(inds_res, 100);

    // Why is inds_res two arrays in one? For optimisation? (inds_res and inds_res+D_HEIGHT.)

    { // 3. the inclusive scan of the condition results
        const unsigned int block_size = 128;
        scanInc<Add<int>,int>(block_size, D_HEIGHT, inds_res, inds_res+D_HEIGHT);
        cudaMemcpy( &filt_size, &inds_res[2*D_HEIGHT - 1],
                    sizeof(int), cudaMemcpyDeviceToHost );
    }
    cudaThreadSynchronize();
    gettimeofday(&t_med2, NULL);
    //cuprintf(inds_res+D_HEIGHT, 100);

    if(VERSION == 2) {
        // version with dummy writes, i.e., writes are performed to
        // global memory in the original order => horible access pattern.
        const unsigned int block_size = 64; //256;
        const unsigned int num_blocks = (D_HEIGHT + block_size - 1) / block_size;
        writeChunkKernelDummy<typename COND::InType><<<num_blocks, block_size>>>
            (d_tr_in, cond_res, inds_res+D_HEIGHT, d_out, D_HEIGHT, D_WIDTH);
    } else {
        // writes are accumulated in shared memory, and performed to
        // global memory in an optimized order (intra-segment order of result).
        const unsigned int SEQ_CHUNK   = D_WIDTH / 32;
        const unsigned int SH_MEM_SIZE = SEQ_CHUNK * 1024 * sizeof(int);
        dim3 block(32, 32, 1);
        dim3 grid ( D_HEIGHT/32, 1, 1);
        //printf("WIDTH: %d, HEIGHT: %d\n\n", d_width, d_height);
        writeChunkKernel<typename COND::InType><<<grid, block, SH_MEM_SIZE>>>
            (d_tr_in, cond_res, inds_res+D_HEIGHT, d_out, D_HEIGHT, SEQ_CHUNK);
    }
    cudaThreadSynchronize();
    gettimeofday(&t_end, NULL);
    //cuprintff(d_out, 100);

    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    if (VERSION == 2)
        printf("Dummy Filter total runtime is: %lu microsecs, from which:\n", elapsed);
    else
        printf("Smart Filter total runtime is: %lu microsecs, from which:\n", elapsed);

    timeval_subtract(&t_diff, &t_med0, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Transposition runs in: %lu microsecs\n", elapsed);

    timeval_subtract(&t_diff, &t_med1, &t_med0);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Map Cond Kernel runs in: %lu microsecs\n", elapsed);

    timeval_subtract(&t_diff, &t_med2, &t_med1);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Scan Addition Kernel runs in: %lu microsecs\n", elapsed);

    timeval_subtract(&t_diff, &t_end, &t_med2);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Global Write Kernel runs in: %lu microsecs\n", elapsed);

    // free resources
    cudaFree(inds_res);
    cudaFree(cond_res);
    cudaFree(d_tr_in );

    return filt_size;
}


template<class DISCR>
typename DISCR::ExpType
multiFilter(    const unsigned int      num_elems,
                const unsigned int      num_hwd_thds,
                typename DISCR::InType* d_in,  // device
                typename DISCR::InType* d_out  // device
) {
    const unsigned int MAX_CHUNK = 384; //256; //384;
    // compute a suitable CHUNK factor and padd the intermediate arrays such
    // that 64 | D_HEIGHT and 32 | D_WIDTH
    const unsigned int D_WIDTH = min( nextMultOf(max(num_elems/num_hwd_thds,1), 32), MAX_CHUNK);  // SEQ CHUNK
    const unsigned int D_HEIGHT= nextMultOf( (num_elems + D_WIDTH - 1) / D_WIDTH, 64 );           // NUM CHUNKS?
    const unsigned int PADD    = nextMultOf(D_HEIGHT*D_WIDTH, 64*D_WIDTH) - num_elems;

    printf("\nD_WIDTH: %d, D_HEIGHT: %d\n\n\n", D_WIDTH, D_HEIGHT);

    struct timeval t_start, t_med0, t_med1, t_med2, t_end, t_diff;
    unsigned long int elapsed;

    typename DISCR::InType *d_tr_in;
    int *cond_res;
    int *gids;
    typename DISCR::ExpType *inds_res;
    typename DISCR::ExpType  filt_size;
    cudaMalloc((void**)&d_tr_in, D_HEIGHT*D_WIDTH*sizeof(typename DISCR::InType));
    cudaMalloc((void**)&cond_res, D_HEIGHT*D_WIDTH*sizeof(int));
    cudaMalloc((void**)&gids, 1000*sizeof(int));
    cudaMalloc((void**)&inds_res, 2*D_HEIGHT*DISCR::cardinal*sizeof(int));

    printf("D_IN IS:\n");
    cuprintf(d_in, 132);
    gettimeofday(&t_start, NULL);
    { // 1. Transpose with padding!
       transposePad<typename DISCR::InType,16>
            (d_in, d_tr_in, D_HEIGHT, D_WIDTH, num_elems, DISCR::padelm);
    }
    printf("D_HEIGHT: %d, D_WIDTH: %d, num_elems: %d, padelwat: %d\n", D_HEIGHT, D_WIDTH, num_elems, DISCR::padelm);
    // D_HEIGHT: 131136, D_WIDTH: 384, num_elems: 50332001, padelwat: 3
    cudaThreadSynchronize();
    printf("D_TR_IN IS:\n");
    cuprintf(d_tr_in, 132);
    gettimeofday(&t_med0, NULL);

    { // 2. The Map Condition Kernel Call
        const unsigned int block_size = 64; //256;
        const unsigned int num_blocks = (D_HEIGHT + block_size - 1) / block_size;
        const unsigned int SH_MEM_MAP  = block_size * DISCR::cardinal * sizeof(int);

        // map the condition
        mapVctKernel<DISCR><<<num_blocks, block_size, SH_MEM_MAP>>>
                (d_tr_in, cond_res, inds_res, gids, D_HEIGHT, D_WIDTH);
    }
    cudaThreadSynchronize();
    cuprintf(gids, 100);
    //cuprintf(inds_res, 50);
    //printf("@@@@@@@@@@@@@@@@@@@@@@@@@ %d @@@@@@@@@@@@@@@@@@\n", inds_res[0].x);
    //cuprintf4(inds_res, D_HEIGHT);
    printf("d_height is: %d\n", D_HEIGHT);
    cuprintf4((MyInt4*)inds_res, 10);
    gettimeofday(&t_med1, NULL);

    { // 3. the inclusive scan of the condition results
        const unsigned int block_size = 128;
        scanInc<typename DISCR::AddExpType,typename DISCR::ExpType>
                (block_size, D_HEIGHT, inds_res, inds_res+D_HEIGHT);

        cudaMemcpy( &filt_size, &inds_res[2*D_HEIGHT - 1],
                    DISCR::cardinal*sizeof(int), cudaMemcpyDeviceToHost );

        filt_size.selSub(DISCR::cardinal, PADD);
//        printf( "sizes: (%d, %d, %d, %d), height: %d, width: %d, pad: %d, num blocks: %d\n\n",
//                filt_size.x, filt_size.y, filt_size.z, filt_size.w, D_HEIGHT, D_WIDTH, PADD, (D_HEIGHT+31)/32 );
    }
    cudaThreadSynchronize();
    printf("@@@@@@@@@@@@@@@@@@@@@@@@@ AFTER SCAN INC\n");
    cuprintf4((MyInt4*)inds_res+D_HEIGHT, 10);
    //cuprintf(inds_res, 50);
    gettimeofday(&t_med2, NULL);

#if 1
    { // 4. the write to global memory part
        // By construction: D_WIDTH  is guaranteed to be a multiple of 32 AND
        //                  D_HEIGHT is guaranteed to be a multiple of 64 !!!
        const unsigned int SEQ_CHUNK   = D_WIDTH / 32;
        printf("SEQ_CHUNK: %u, DISCRcaqrd: %u\n\n", SEQ_CHUNK, DISCR::cardinal);
        const unsigned int SH_MEM_SIZE = 1024 * sizeof(int) * max(SEQ_CHUNK,DISCR::cardinal);

        dim3 block(32, 32, 1);
        dim3 grid ( D_HEIGHT/32, 1, 1);
        writeMultiKernel<DISCR><<<grid, block, SH_MEM_SIZE>>>
            (d_tr_in, cond_res, inds_res+D_HEIGHT, d_out, D_HEIGHT, num_elems, SEQ_CHUNK);
    }
#else
    { // 4. the write to global memory part
        // By construction: D_WIDTH  is guaranteed to be a multiple of 32 AND
        //                  D_HEIGHT is guaranteed to be a multiple of 64 !!!
        const unsigned int SEQ_CHUNK   = D_WIDTH / 32;
        printf("SEQ_CHUNK: %u, DISCRcard: %u\n\n", SEQ_CHUNK, DISCR::cardinal);
        const unsigned int SH_MEM_SIZE = 1024 * sizeof(int) * SEQ_CHUNK + 32*33*sizeof(int); // PUT CORRECT TYPES!

        dim3 block(32, 32, 1);
        dim3 grid ( D_HEIGHT/32, 1, 1);
        writeMultiKernelOpt<Mod4Opt><<<grid, block, SH_MEM_SIZE>>>
            (d_in, cond_res, inds_res+D_HEIGHT, d_out, D_HEIGHT, num_elems, SEQ_CHUNK);
    }
#endif
    cudaThreadSynchronize();
    gettimeofday(&t_end, NULL);

    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Multi-Filter total runtime is: %lu microsecs, from which:\n", elapsed);

    timeval_subtract(&t_diff, &t_med0, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Transposition runs in: %lu microsecs\n", elapsed);

    timeval_subtract(&t_diff, &t_med1, &t_med0);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Map Cond Kernel runs in: %lu microsecs\n", elapsed);

    timeval_subtract(&t_diff, &t_med2, &t_med1);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Scan Addition Kernel runs in: %lu microsecs\n", elapsed);

    timeval_subtract(&t_diff, &t_end, &t_med2);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Global Write Kernel runs in: %lu microsecs\n", elapsed);

    // free resources
    cudaFree(inds_res);
    cudaFree(cond_res);
    cudaFree(d_tr_in );

    return filt_size;

}

template<class DISCR>
typename DISCR::ExpType
sgmMultiFilter( const unsigned int      num_elems,
                      unsigned int      num_segments,
                const unsigned int      num_hwd_thds,
                typename DISCR::InType* d_in,  // device
                unsigned int*     flags, // device
                typename DISCR::InType* d_out  // device
) {
    const unsigned int MAX_CHUNK = 384; //256; //384;
    // compute a suitable CHUNK factor and padd the intermediate arrays such
    // that 64 | D_HEIGHT and 32 | D_WIDTH
    const unsigned int D_WIDTH = min( nextMultOf(max(num_elems/num_hwd_thds,1), 32), MAX_CHUNK);  // SEQ CHUNK
    const unsigned int D_HEIGHT= nextMultOf( (num_elems + D_WIDTH - 1) / D_WIDTH, 64 );
    const unsigned int PADD    = nextMultOf(D_HEIGHT*D_WIDTH, 64*D_WIDTH) - num_elems;

    printf("PADD IS: %d\n", PADD);

    struct timeval t_start, t_med0, t_med1, t_med2, t_end, t_diff;
    unsigned long int elapsed;

    typename DISCR::InType *d_tr_in;
    int *cond_results;
    unsigned int* flags_tr;
    typename DISCR::ExpType *chunk_counters;
    typename DISCR::ExpType *segment_counters;
    unsigned int *segment_ids, *segment_ids_tr;
    unsigned int *chunk_flags;
    typename DISCR::ExpType  filt_size;

    cudaMalloc((void**)&d_tr_in,      D_HEIGHT * D_WIDTH * sizeof(typename DISCR::InType));
    cudaMalloc((void**)&cond_results, D_HEIGHT * D_WIDTH * sizeof(int));
    cudaMalloc((void**)&flags_tr,     D_HEIGHT * D_WIDTH * sizeof(int));

    // Shouldn't DISCR::cardinal*sizeof(int) be sizeof(DISCR::ExpType)?
    cudaMalloc((void**)&chunk_counters,   2 * (D_HEIGHT     * DISCR::cardinal * sizeof(int)));
    cudaMalloc((void**)&segment_counters, 2 * (num_segments * DISCR::cardinal * sizeof(int)));
    cudaMalloc((void**)&segment_ids,    D_HEIGHT * D_WIDTH * sizeof(int)); // was *4. mistake, yes?
    cudaMalloc((void**)&segment_ids_tr, D_HEIGHT * D_WIDTH * sizeof(int)); // was *4. mistake, yes?

    cudaMemset((void*)chunk_counters, 0, 2 * (D_HEIGHT * DISCR::cardinal * sizeof(int)));
    cudaMemset((void*)cond_results, 0, D_HEIGHT * D_WIDTH * sizeof(int));
    cudaMemset((void*)segment_counters, 0, 2 * (num_segments * DISCR::cardinal * sizeof(int)));

    {
        const unsigned int block_size = 64; //TODO FIXME: Should we use this block size?
        // Prepare segment_ids (flags scanned over to 11112222333444 format...)
        // Note: It's 1-indexed. Values subtracted by 1 when necessary during use.
        scanInc<Add<unsigned int>, unsigned int>
            ((unsigned int)block_size, (unsigned long)num_elems, flags, segment_ids);
    }

    cudaThreadSynchronize();
    
    printf("D_IN IS:\n");
    cuprintf(d_in, 128);

    gettimeofday(&t_start, NULL);
    { // 1. Transpose with padding!
       transposePad<typename DISCR::InType, 16>
            (d_in, d_tr_in, D_HEIGHT, D_WIDTH, num_elems, DISCR::padelm);
       // FIXME TODO: Flags should be transposed the same way as the regular elements.
       // Otherwise indexing needs to be done in transposed manner in-place.
       transposePad<unsigned int, 16>
            (flags, flags_tr, D_HEIGHT, D_WIDTH, num_elems, 0);
       transposePad<unsigned int, 16>
            (segment_ids, segment_ids_tr, D_HEIGHT, D_WIDTH, num_elems, 0);
    }
    //printf("D_HEIGHT: %d, D_WIDTH: %d, num_elems: %d, padelwat: %d\n", D_HEIGHT, D_WIDTH, num_elems, DISCR::padelm);
    // D_HEIGHT: 131136, D_WIDTH: 384, num_elems: 50332001, padelwat: 3
    // D_HEIGHT: 64, D_WIDTH: 32, num_elems: 1024, padelwat: 3

    cudaThreadSynchronize();

    //printf("FLAGS ARE:\n");
    //cuprintf((int*)flags, 1024);
    printf("SEG IDS ARE (num_elems: %d):\n", num_elems);
    cuprintf((int*)segment_ids, 32*64);
    printf("SEG IDS_TR ARE (num_elems: %d):\n", num_elems);
    cuprintf((int*)segment_ids_tr, 32*64);
    printf("FLAGS ARE (num_elems: %d):\n", num_elems);
    cuprintf((int*)flags, 32*64);
    printf("FLAGS_TR ARE (num_elems: %d):\n", num_elems);
    cuprintf((int*)flags_tr, 32*64);

    printf("D_TR_IN IS:\n");
    cuprintf(d_tr_in, 128);

    //printf("SEG_IDS_TR IS:\n");
    //cuprintf((int*)segment_ids_tr, 128);

    printf("COND_RES BEFORE KERNEL IS:\n");
    cuprintf(cond_results, 128);

    printf("SEG_INDS_RES (SEGMENT COUNTERS) BEFORE KERNEL CALL IS (num segments: %d):\n", num_segments);
    cuprintf4(segment_counters, 32);

    gettimeofday(&t_med0, NULL);

    { // 2. The Map Condition Kernel Call
        const unsigned int block_size = 64; //256;
        const unsigned int num_blocks = (D_HEIGHT + block_size - 1) / block_size;
        const unsigned int SH_MEM_MAP  = (block_size + num_segments) * DISCR::cardinal * sizeof(int);

        cudaMalloc((void**)&chunk_flags, 2*block_size*sizeof(int));

        // map the condition
        sgmMapVctKernel<DISCR><<<num_blocks, block_size, SH_MEM_MAP>>>
                (d_tr_in, cond_results, chunk_counters, segment_counters, flags_tr, segment_ids_tr, chunk_flags, D_HEIGHT, D_WIDTH);
    }
    cudaThreadSynchronize();

    printf("SEG_INDS_RES (SEGMENT COUNTERS, %p) AFTER KERNEL CALL IS:\n", segment_counters);
    cuprintf4(segment_counters, num_segments);
    printf("COND_RES IS:\n");
    cuprintf(cond_results, 512);
    printf("INDS_RES (CHUNK COUNTERS) IS:\n");
    cuprintf4(chunk_counters, D_HEIGHT);

    const unsigned int block_size = 64; //256;
    const unsigned int num_blocks = (D_HEIGHT + block_size - 1) / block_size;
    printf("CHUNK_FLAGS ARE:\n");
    cuprintf((int*)chunk_flags, 64);

    {
        // TODO FIXME: This block size should be same as in map. Use a #define or something?
        const unsigned int block_size = 64;
        // Prepare chunk_ids (flags scanned over to 11112222333444 format...)
        // Note: It's 1-indexed. Values subtracted by 1 when necessary during use.
        scanInc<Add<unsigned int>, unsigned int>
            ((unsigned int)block_size, (unsigned long)block_size, chunk_flags, chunk_flags + block_size);
    }

    printf("CHUNK_FLAGS IDS ARE:\n");
    cuprintf((int*)(chunk_flags + 64), 64);

    gettimeofday(&t_med1, NULL);

    { // 3. the inclusive scan of the condition results
        const unsigned int block_size = 128;
        sgmScanInc<typename DISCR::AddExpType,typename DISCR::ExpType>
                (block_size, D_HEIGHT, chunk_counters, (int*)chunk_flags, chunk_counters + D_HEIGHT);
        //scanInc<typename DISCR::AddExpType,typename DISCR::ExpType>
                //(block_size, num_segments, segment_counters, segment_counters + num_segments);

        // TODO FIXME: Fix magic number 64. It's block size from when block_flags is scanned to 11122233 format.
        unsigned int block_size_new = 64;
        const unsigned int num_blocks = (D_HEIGHT + block_size_new - 1) / block_size_new;
        const unsigned int num_chunks = (int)ceil((float)num_elems / (float)D_WIDTH);
        mapAggregateCounters<DISCR><<<num_blocks, block_size_new>>>
                (segment_counters, segment_ids, chunk_flags, chunk_counters + D_HEIGHT, D_WIDTH, num_chunks);

        cudaMemcpy( &filt_size, &chunk_counters[2*D_HEIGHT - 1],
                    DISCR::cardinal*sizeof(int), cudaMemcpyDeviceToHost );

        filt_size.selSub(DISCR::cardinal, PADD);
    }

    printf("SEG_INDS_RES (AGGREGATED COUNTERS, %p) AFTER SCAN KERNEL CALL IS:\n", segment_counters + num_segments);
    cuprintf4(segment_counters, num_segments);
    //cuprintf4(segment_counters + num_segments, num_segments);
    printf("INDS_RES (CHUNK COUNTERS) AFTER SCAN KERNEL IS:\n");
    cuprintf4(chunk_counters + D_HEIGHT, D_HEIGHT);
    printf("NUM SEGMENTS IS: %d\n", num_segments);

    printf("FILT_SIZE IS: %d, %d, %d, %d\n", filt_size.x, filt_size.y, filt_size.z, filt_size.w);

    cudaThreadSynchronize();
    gettimeofday(&t_med2, NULL);


    { // 4. the write to global memory part
        // By construction: D_WIDTH  is guaranteed to be a multiple of 32 AND
        //                  D_HEIGHT is guaranteed to be a multiple of 64 !!!
        const unsigned int SEQ_CHUNK   = D_WIDTH / 32;
        printf("SEQ_CHUNK: %u, DISCRcaqrd: %u\n\n", SEQ_CHUNK, DISCR::cardinal);
        const unsigned int SH_MEM_SIZE = 1024 * sizeof(int) * max(SEQ_CHUNK,DISCR::cardinal);

        dim3 block(32, 32, 1);
        dim3 grid ( D_HEIGHT/32, 1, 1);
        writeMultiKernel<DISCR><<<grid, block, SH_MEM_SIZE>>>
            (d_tr_in, cond_results, chunk_counters + D_HEIGHT, d_out, D_HEIGHT, num_elems, SEQ_CHUNK);
    }
    cudaThreadSynchronize();
    gettimeofday(&t_end, NULL);

    /*

    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Multi-Filter total runtime is: %lu microsecs, from which:\n", elapsed);

    timeval_subtract(&t_diff, &t_med0, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Transposition runs in: %lu microsecs\n", elapsed);

    timeval_subtract(&t_diff, &t_med1, &t_med0);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Map Cond Kernel runs in: %lu microsecs\n", elapsed);

    timeval_subtract(&t_diff, &t_med2, &t_med1);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Scan Addition Kernel runs in: %lu microsecs\n", elapsed);

    timeval_subtract(&t_diff, &t_end, &t_med2);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Global Write Kernel runs in: %lu microsecs\n", elapsed);

    */

    // free resources
    cudaFree(d_tr_in);
    cudaFree(cond_results);
    cudaFree(flags_tr);

    cudaFree(chunk_counters);
    cudaFree(segment_counters);
    cudaFree(segment_ids);
    cudaFree(segment_ids_tr);

    cudaFree(chunk_flags);

    return filt_size;
}

#endif //PAR_BB_HOST
