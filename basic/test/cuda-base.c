/* coding: utf-8 */
#include<stdio.h>
#include<stdbool.h>
#include<cuda_runtime.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    madd_error_keep_print = true;

    int n_cuda = Madd_N_cuda_GPU();
    printf("cuda device: %d\n", n_cuda);

    printf("List Devices:\n");
    int i_dev;
    size_t free_byte, total_byte;
    struct cudaDeviceProp dp;
    Madd_cuda_Device_Properties mdp = Madd_cuda_Get_Device_Property();
    for (i_dev=0; i_dev<mdp.n_device; i_dev++){
        dp = mdp.devices[i_dev];
        printf("\n\n=== Device %d ===\n", i_dev);
        printf("Device name:\t\t\t%s\n", dp.name);
        printf("Total mem:\t\t\t%zd bytes | %.1f kb | %.1f Mb | %.1f Gb\n", dp.totalGlobalMem, dp.totalGlobalMem/1024., dp.totalGlobalMem/(1024.*1024.), dp.totalGlobalMem/(1024.*1024.*1024.));
        printf("shared mem per block:\t\t%zd bytes | %.1f kb\n", dp.sharedMemPerBlock, dp.sharedMemPerBlock/1024.);
        printf("registers per block:\t\t%d\n", dp.regsPerBlock);
        printf("warp size:\t\t\t%d\n", dp.warpSize);
        printf("mem pitch:\t\t\t%zd bytes | %.1f kb | %.1f Mb | %.1f Gb\n", dp.memPitch, dp.memPitch/1024., dp.memPitch/(1024.*1024.), dp.memPitch/(1024.*1024.*1024.));
        printf("max threads per block:\t\t%d > (%d, %d, %d)\n", dp.maxThreadsPerBlock, dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
        printf("max grid size:\t\t\t(%d, %d, %d)\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);
        printf("total constant mem:\t\t%zd bytes | %.1f kb\n", dp.totalConstMem, dp.totalConstMem/1024.);
        printf("computation capability:\t\t%d.%d\n", dp.major, dp.minor);
        //printf("clock rate:\t\t\t%d kHz | %.1f MHz | %.2f GHz\n", dp.clockRate, dp.clockRate/1.e3, dp.clockRate/1.e6);
        printf("texture alignment:\t\t%zd\n", dp.textureAlignment);
        /*if (dp.deviceOverlap){
            printf("device overlap:\t\t\tTrue\n");
        }else{
            printf("device overlap:\t\t\tFalse");
        }*/
        printf("multi processorCount:\t\t%d\n", dp.multiProcessorCount);
        /*if (dp.kernelExecTimeoutEnabled){
            printf("kernel execution time out:\tenabled\n");
        }else{
            printf("kernel execution time out:\tdisabled\n");
        }*/
        if (dp.integrated){
            printf("integrated GPU:\t\t\tTrue\n");
        }else{
            printf("integrated GPU:\t\t\tFalse\n");
        }
        if (dp.canMapHostMemory){
            printf("Map Host Mem:\t\t\tTrue\n");
        }else{
            printf("Map Host Mem:\t\t\tFalse\n");
        }
        //printf("compute mode:\t%d\n", computeMode);
        printf("max texture 1d-mem:\t\t%d bytes | %.2f kb\n", dp.maxTexture1D, dp.maxTexture1D/1.e3);
        printf("max texture 2d-mem:\t\t(%d, %d) bytes | (%.1f, %.1f) kb\n", dp.maxTexture2D[0], dp.maxTexture2D[1], dp.maxTexture2D[0]/1.e3, dp.maxTexture2D[1]/1.e3);
        printf("max texture 3d-mem:\t\t(%d, %d, %d) bytes | (%.1f, %.1f, %.1f) kb\n", dp.maxTexture3D[0], dp.maxTexture3D[1], dp.maxTexture3D[2], dp.maxTexture3D[0]/1.e3, dp.maxTexture3D[1]/1.e3, dp.maxTexture3D[2]/1.e3);
        //printf("max texture 2d-array:\t(%d, %d, %d)\n", dp.maxTexture2DArray[0], dp.maxTexture2DArray[1], dp.maxTexture2DArray[2]);
        if (dp.concurrentKernels){
            printf("kernels concurrent available:\tTrue\n");
        }else{
            printf("kernels concurrent available:\tFalse\n");
        }
        /* available mem */
        Madd_cuda_Get_Device_Mem(i_dev, &free_byte, &total_byte);
        printf("\nfree mem:\t%zd bytes | %.1f kb | %.1f Mb | %.1f Gb\n", free_byte, free_byte/1024., free_byte/(1024.*1024.), free_byte/(1024.*1024.*1024.));
        printf("total mem:\t%zd bytes | %.1f kb | %.1f Mb | %.1f Gb\n", total_byte, total_byte/1024., total_byte/(1024.*1024.), total_byte/(1024.*1024.*1024.));
    }
    return 0;
}