#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cuda.h>


#ifndef MC_CONFIG_ID
  #include "mc_config1.cu"   // default config
  #define MC_CONFIG_ID 1
#else // choose config (set via cmake)
  #if MC_CONFIG_ID==1
    #include "mc_config1.cu"
  #elif MC_CONFIG_ID==2
    #include "mc_config2.cu"
  #elif MC_CONFIG_ID==3
    #include "mc_config3.cu"
  #endif
#endif



#define CHECK_CUDA(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

/*only check if kernel start is valid*/
#define CHECK_CUDA_KERNEL(...) __VA_ARGS__;CHECK_CUDA(cudaGetLastError())

// I know it is not a Object oriented style
const int arrCnt = 4;
const int minArrUsed = 2;
const int maxArrUsed = 4;

typedef char* (threadArr_t)[arrCnt];
__device__ threadArr_t * a;

uint64_t * clockTicks;
uint32_t * allocatedChunksCount;

ScatterAllocator *mmc = NULL;

// standard nvidia allocator
__global__ void cuNewCreateArrayPointer(int size){
  a = new threadArr_t[size];
}

__global__ void cuNewFreeArrayPointer(){
  delete [] a;
}

__global__ void cuNewArray(unsigned int chunkSize,
                           unsigned int maxChunkCountPerAlloc,
                           unsigned int arrIdx,
                           uint64_t *clockTicks,
                           uint32_t *allocCount){
  // allocate just some "random" requested size (in multipes of chunksize)
  unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int chunkAlloc = 
                  (1103515245 * arrIdx + id) % maxChunkCountPerAlloc + 1;
  unsigned int size = chunkSize*chunkAlloc;
  clock_t start = clock64();
  a[id][arrIdx] = new char[size];
  clock_t end = clock64();
  allocCount[id] = chunkAlloc;
  clockTicks[id] = end - start;
}

__global__ void cuFreeArray(unsigned int arrIdx){
   unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
   delete [] a[id][arrIdx];  
}


// mallocMC
__global__ void mmcCreateArrayPointer(int size, ScatterAllocator::AllocatorHandle mMC){
  a = (threadArr_t*) mMC.malloc(sizeof(threadArr_t*) * size);
}

__global__ void mmcNewFreeArrayPointer(ScatterAllocator::AllocatorHandle mMC){
  mMC.free(a);
}

__global__ void mmcNewArray(unsigned int chunkSize,
                           unsigned int maxChunkCountPerAlloc,
                           unsigned int arrIdx,
                           uint64_t *clockTicks,
                           uint32_t *allocCount,
                           ScatterAllocator::AllocatorHandle mMC){
  // allocate just some "random" requested size (in multipes of chunksize)
  unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int chunkAlloc = 
                  (1103515245 * arrIdx + id) % maxChunkCountPerAlloc + 1;
  unsigned int size = chunkSize*chunkAlloc;
  clock_t start = clock64();
  a[id][arrIdx] = (char*) mMC.malloc(size*sizeof(char));
  clock_t end = clock64();
  allocCount[id] = chunkAlloc;
  clockTicks[id] = end - start;
}

__global__ void mmcFreeArray(unsigned int arrIdx, ScatterAllocator::AllocatorHandle mMC){
   unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
   mMC.free(a[id][arrIdx]);
}

// host functions
int gpuInit(bool useMallocMC){
  int ccMajor = 0;
  cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, 0);
  int ccMinor = 0;
  cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, 0);

  if( ccMajor < 2 ) {
    printf("Error: Compute Capability >= 2.0 required. (is %d.%d)\n",
                  ccMajor, ccMinor);
    return -1;
  }
  CHECK_CUDA(cudaSetDevice(0));
  size_t free, total;
  CHECK_CUDA(cudaMemGetInfo(&free, &total));
  if(useMallocMC){
    size_t allocMem = (free *7) / 8;  // dont allocate/reserve all memory
    mmc = new ScatterAllocator(allocMem);
    return allocMem;
  }else{
    return free;
  }
}

void gpuFree(unsigned int arrIdx, int gs, int bs){
  if(mmc != NULL){
    CHECK_CUDA_KERNEL(mmcFreeArray<<<gs, bs>>>(arrIdx, *mmc));
  }else{
    CHECK_CUDA_KERNEL(cuFreeArray<<<gs, bs>>>(arrIdx));
  }
}

int gpuAlloc(unsigned int chunkSize, unsigned int maxChunkCountPerAlloc, 
                  unsigned int arrIdx, int gs, int bs, uint64_t *clockRes){
  if(mmc != NULL){
    CHECK_CUDA_KERNEL(mmcNewArray<<<gs, bs>>>(chunkSize,
                     maxChunkCountPerAlloc, arrIdx,
                     clockTicks, allocatedChunksCount, *mmc));
  }else{
    CHECK_CUDA_KERNEL(cuNewArray<<<gs, bs>>>(chunkSize,
                     maxChunkCountPerAlloc, arrIdx,
                     clockTicks, allocatedChunksCount));
  }
  int size = gs * bs;
  thrust::device_ptr<uint32_t> thAllocatedChunksCount =
            thrust::device_pointer_cast(allocatedChunksCount);
  //thrust::device_vector<uint64_t> ct(thClockTicks, thClockTicks+size);

  if(clockRes != NULL){
    thrust::device_ptr<uint64_t> thClockTicks = thrust::device_pointer_cast(clockTicks);
    uint64_t clockSum = thrust::reduce(thClockTicks, thClockTicks+size);
    uint64_t clockMin = *thrust::min_element(thClockTicks, thClockTicks + size);
    uint64_t clockMax = *thrust::max_element(thClockTicks, thClockTicks + size);

    clockRes[0] = clockMin;
    clockRes[1] = clockSum / size;
    clockRes[2] = clockMax;
  }
  return thrust::reduce(thAllocatedChunksCount, 
                        thAllocatedChunksCount + size);
}


void run(int chunkSize, int maxChunkCountPerAlloc, int warmUpRounds, int testRounds,
         int gridSize, int blockSize, ScatterAllocator *mmc, FILE *fp){        
  fprintf(fp, "%d %d %d %d %d %d %d %d %d %d\n", chunkSize, maxChunkCountPerAlloc,
               (mmc==NULL) ? 0 : 1, 
               warmUpRounds, testRounds, gridSize, blockSize, 
               arrCnt, minArrUsed, maxArrUsed); 
  // init round (request Memory)
  uint32_t size = blockSize * gridSize;
  CHECK_CUDA (cudaMalloc(&clockTicks, sizeof(uint64_t) *size));
  CHECK_CUDA (cudaMalloc(&allocatedChunksCount, sizeof(uint32_t) *size));
  if(mmc != NULL){
    CHECK_CUDA_KERNEL(mmcCreateArrayPointer<<<1,1>>>(size, *mmc));
  } else {
    CHECK_CUDA_KERNEL(cuNewCreateArrayPointer<<<1,1>>>(size));
  }

  // initial state of arrays
  int arrAllocated[arrCnt];  // stores sum of allocated chunks for each arrayNumber
  uint64_t clockTimingSum[arrCnt][3];
  uint64_t chunkRequestSum[arrCnt];
  uint64_t chunkAllocatedSum[arrCnt];
  int arrAllocRequestCount[arrCnt];
  for(int i=0; i<arrCnt; i++){
    arrAllocated[i] = 0;
    chunkRequestSum[i] = 0;
    chunkAllocatedSum[i] = 0;
    clockTimingSum[i][0] = 0;
    clockTimingSum[i][1] = 0;
    clockTimingSum[i][2] = 0;
    arrAllocRequestCount[i] = 0; 
  }

  // warm up 
  int arrAllocatedCnt = 0;
  for(int i=0; i<warmUpRounds; i++){
    int arrIdx = rand() % arrCnt;
    // update arrIdx, until alloc/free respects fill boundaries
    while( (arrAllocatedCnt<minArrUsed && arrAllocated[arrIdx]!=0 ) ||
           (arrAllocatedCnt>=maxArrUsed && arrAllocated[arrIdx]==0) ){
      arrIdx = rand() % arrCnt;
    }
    if(arrAllocated[arrIdx]){
      gpuFree(arrIdx, gridSize, blockSize);
      arrAllocated[arrIdx] = 0;
      arrAllocatedCnt--;
    } else {
      arrAllocated[arrIdx] = gpuAlloc(chunkSize, maxChunkCountPerAlloc,
                                     arrIdx, gridSize, blockSize,
                                     NULL);
      arrAllocatedCnt++;
    }
  }

  // repeat until number of allocation calls equals testRounds
  int validAllocCnt = 0;
  uint64_t clockTiming[3];
  while(validAllocCnt<testRounds){
    int arrIdx = rand() % arrCnt;
    // update arrIdx, until alloc/free respects fill boundaries
    while( (arrAllocatedCnt<=minArrUsed && arrAllocated[arrIdx]) ||
           (arrAllocatedCnt>=maxArrUsed && !arrAllocated[arrIdx])){
      arrIdx = rand() % arrCnt;
    }
    if(arrAllocated[arrIdx] != 0){
      gpuFree(arrIdx, gridSize, blockSize);
      arrAllocated[arrIdx] = 0;
      arrAllocatedCnt--;
    } else {
      int allocChunkSum = 0;
      for(int i=0; i<arrCnt; i++){
        allocChunkSum += arrAllocated[i];
      }
      arrAllocated[arrIdx] = gpuAlloc(chunkSize, maxChunkCountPerAlloc, 
                                    arrIdx, gridSize, blockSize,
                                    clockTiming);

      chunkAllocatedSum[arrAllocatedCnt] += allocChunkSum;
      chunkRequestSum[arrAllocatedCnt] += arrAllocated[arrIdx];
      clockTimingSum[arrAllocatedCnt][0] += clockTiming[0];
      clockTimingSum[arrAllocatedCnt][1] += clockTiming[1];
      clockTimingSum[arrAllocatedCnt][2] += clockTiming[2];
      arrAllocRequestCount[arrAllocatedCnt]++;
      // write single run infos to file
      /*fprintf(fp, "%d %d %d %s", arrAllocatedCnt, allocChunkSum,
                                 arrAllocated[arrIdx], clockStr); */
      arrAllocatedCnt++;
      validAllocCnt++;
    }
  }
  // write averaged infos to file:
  for(int i=minArrUsed; i<maxArrUsed; i++){
    int cnt = arrAllocRequestCount[i];
    if(cnt != 0){
      fprintf(fp, "%d %d %lu %lu %lu %lu %lu\n", i, cnt, chunkAllocatedSum[i]/cnt,
            chunkRequestSum[i]/cnt, clockTimingSum[i][0]/cnt,
            clockTimingSum[i][1] / cnt, clockTimingSum[i][2] /cnt);
    } else {
      fprintf(fp, "%d 0 0 0 0 0 0\n", i);
    }        
  }
  // free memory of GPU
  for(int i=0; i<arrCnt; i++){
    if(arrAllocated[i] != 0){
      gpuFree(i, gridSize, blockSize);
    }
  }
  CHECK_CUDA(cudaFree(clockTicks));
  CHECK_CUDA(cudaFree(allocatedChunksCount));
  CHECK_CUDA_KERNEL(cuNewFreeArrayPointer<<<1,1>>>());
}

int main(int argc, char *argv[]){
  int chunkSize = 4096;
  int maxChunkCountPerAlloc = 1;
  int warmUpRounds = 200; // maybe increase
  int testRounds = 500;
  bool useMallocMC = false;

  if(argc == 1){
    printf("Usage: useMC chunkSize maxChunkSizePerAlloc\n");
    return 0;
  }
  if(argc > 1){
    useMallocMC = (atoi(argv[1]) != 0);
  }
  if(argc > 2){
    chunkSize = atoi(argv[2]);
  }
  if(argc > 3){
    maxChunkCountPerAlloc = atoi(argv[3]);
  }

  srand (time(NULL));
  int freeMemBytes = gpuInit(useMallocMC);
  printf("Useable Memory: %d Bytes\n", freeMemBytes);
  int maxChunks = freeMemBytes / (sizeof(char) *chunkSize );
  int maxThreads = maxChunks / (arrCnt * maxChunkCountPerAlloc); 
  char fname[255];
  if(useMallocMC){
    sprintf(fname, "mallocMC(%d)_%d_%d.txt", MC_CONFIG_ID, chunkSize, maxChunkCountPerAlloc);
  }else{
   sprintf(fname, "new_%d_%d.txt", chunkSize, maxChunkCountPerAlloc);
  }
  const int sizeCnt = 44;
  int gs [sizeCnt] = {1  , 1  , 1  , 1  , 1  , 1  , 1  , 1  , 
                         1  , 1  , 1  , 1  , 1  , 1  , 1  , 1  ,
                         1  , 1  , 1  , 1  , 4  , 5  , 6  , 7  ,
                         8  , 10 , 12 , 14 , 16 , 20 , 24 , 28 ,
                         32 , 40 , 48 , 56 , 64 , 80 , 96 , 128,
                         160, 192, 224, 256};
  int bs [sizeCnt] = {2  , 4  , 6  , 8  , 12 , 16 , 20 , 24 ,
                         28 , 32 , 40 , 48 , 56 , 64 , 80 , 96 ,
                         128, 160, 192, 224, 64 , 64 , 64 , 64 ,
                         64 , 64 , 64 , 64 , 64 , 64 , 64 , 64 ,
                         64 , 64 , 64 , 64 , 64 , 64 , 64 , 64 ,
                         64 , 64 , 64 , 64 };

  
  FILE *fp = fopen(fname, "w");
  // write header of file
  fprintf(fp, "# Contains multiple averaged testruns\n");
  fprintf(fp, "# Test starts with: chunkSize, maxChunkCountPerAlloc,\n");
  fprintf(fp, "#                   useMC, warmUpRounds, testRounds, gridSize,\n");
  fprintf(fp, "#                   blockSize, arrCnt, minArrUsed, maxArrUsed\n");
  fprintf(fp, "# Run lines:  arrUsed, cnt (over all cases), avg alloc mem,\n");
  fprintf(fp, "#             avg min, avg mean, avg max\n");


  for(int i=0; i<sizeCnt; i++){
    int threadCount = gs[i] * bs[i];
    if(threadCount < maxThreads){
      printf("%d ", threadCount);
      fflush(stdout);
      run(chunkSize, maxChunkCountPerAlloc, warmUpRounds, testRounds, 
          gs[i], bs[i], mmc, fp);
    }
  }
  printf("\n");
  fclose(fp); 
  if(mmc != NULL){
    delete mmc;
  }
  cudaDeviceReset();
  return 0; 
}

