#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// Real galaxy arrays
float *ra_real, *decl_real;
//Number of real galaxies
int    NoofReal;
// Simulated galaxy arrays
float *ra_sim, *decl_sim;
// Number of simulated galaxies
int    NoofSim;
// Number of threads per block
int threadsPerBlock = 512;
// Unit Conversion
const float ARCMIN_TO_RAD = (M_PI / 180.0f) / 60.0f;
const float RAD_TO_DEG = 180.0f / M_PI;
// Bin cofiguration
const float BIN_SIZE = 0.25f;
const int NUM_BINS = (int)(90.0f / BIN_SIZE);
//Real - Real Histogram
long long histogramDD[NUM_BINS] = {0};
//Real - Simulated Histogram
long long histogramDR[NUM_BINS] = {0};
// Simulated  - Simulated Histogram
long long histogramRR[NUM_BINS] = {0};
// Omega values
float omega[NUM_BINS] = {0.0f};

// Calculate the angular separation between two galaxies on the GPU
__device__ float calculateAngularSeparationKernel(float raOneRad, float declOneRad, float raTwoRad, float declTwoRad) {
  float temp = sinf(declOneRad) * sinf(declTwoRad) +
               cosf(declOneRad) * cosf(declTwoRad) *
               cosf(raOneRad - raTwoRad);

  if (temp > 1.0f) temp = 1.0f;
  if (temp < -1.0f) temp = -1.0f;

  return acosf(temp) * RAD_TO_DEG;
}
// Calculate histograms on the GPU
__global__ void fillBinsOptimizedKernel(float *ra_real, float *decl_real, int NoofReal, float *ra_sim, float *decl_sim, int NoofSim, long long *histogramDD, long long *histogramDR, long long *histogramRR) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NoofReal) {
        float raRealRad = ra_real[i] * ARCMIN_TO_RAD;
        float declRealRad = decl_real[i] * ARCMIN_TO_RAD;

        for (int j = 0; j < NoofReal; j++) {
            float raOtherRad = ra_real[j] * ARCMIN_TO_RAD;
            float declOtherRad = decl_real[j] * ARCMIN_TO_RAD;

            float resDeg = calculateAngularSeparationKernel(raRealRad, declRealRad, raOtherRad, declOtherRad);
            int binIndex = (int)(resDeg / BIN_SIZE);
            atomicAdd((unsigned long long int*)&histogramDD[binIndex], 1ULL);
        }

        for (int j = 0; j < NoofSim; j++) {
            float raSimRad = ra_sim[j] * ARCMIN_TO_RAD;
            float declSimRad = decl_sim[j] * ARCMIN_TO_RAD;

            float resDeg = calculateAngularSeparationKernel(raRealRad, declRealRad, raSimRad, declSimRad);
            int binIndex = (int)(resDeg / BIN_SIZE);
            atomicAdd((unsigned long long int*)&histogramDR[binIndex], 1ULL);
        }
    }

    if (i < NoofSim) {
        float raSimRad = ra_sim[i] * ARCMIN_TO_RAD;
        float declSimRad = decl_sim[i] * ARCMIN_TO_RAD;

        for (int j = 0; j < NoofSim; j++) {
            float raOtherRad = ra_sim[j] * ARCMIN_TO_RAD;
            float declOtherRad = decl_sim[j] * ARCMIN_TO_RAD;

            float resDeg = calculateAngularSeparationKernel(raSimRad, declSimRad, raOtherRad, declOtherRad);
            int binIndex = (int)(resDeg / BIN_SIZE);
            atomicAdd((unsigned long long int*)&histogramRR[binIndex], 1ULL);
        }
    }
}

// Calculate Omega in CPU
int calculateOmega() {
    for (int i = 0; i < NUM_BINS; i++) {
        if (histogramRR[i] != 0) {
            omega[i] = (float)(histogramDD[i] - 2*histogramDR[i] + histogramRR[i]) / histogramRR[i];
        } else {
            omega[i] = 0.0f;
        }
    }
    return 0;
}

int printResults(){
    int count = 15;
    long long histogramDDsum = 0;
    long long histogramDRsum = 0;
    long long histogramRRsum = 0;


    printf("First %d bins:\n", count);
    printf("HistogramDD:\n");
    for (int i = 0; i < NUM_BINS; i++){
        if (i < count){
            printf("Bin %d: %d\n", i, histogramDD[i]);
        }
        histogramDDsum += histogramDD[i];
    }
    printf("Histogram DD Sum: %lld\n", histogramDDsum);

    printf("HistogramDR:\n");
    for (int i = 0; i < NUM_BINS; i++){
        if (i < count){
            printf("Bin %d: %d\n", i, histogramDR[i]);
        }
        histogramDRsum += histogramDR[i];
    }
    printf("Histogram DR Sum: %lld\n", histogramDRsum);

    printf("HistogramRR:\n");
    for (int i = 0; i < NUM_BINS; i++){
        if (i < count){
            printf("Bin %d: %d\n", i, histogramRR[i]);
        }
        histogramRRsum += histogramRR[i];
    }
    printf("Histogram RR Sum: %lld\n", histogramRRsum);

    printf("Omega:\n");
    for (int i = 0; i < 10; i++) printf("Bin %d: %f\n", i, omega[i]);
    return 0;
}


int main(int argc, char *argv[]){
   int    i;
   int    readdata(char *argv1, char *argv2);
   int    getDevice(int deviceno);
   long long histogramDRsum, histogramDDsum, histogramRRsum;
   double w;
   double start, end, kerneltime;
   struct timeval _ttime;
   struct timezone _tzone;
   cudaError_t myError;

   FILE *outfil;

    if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

    if ( getDevice(0) != 0 ) return(-1);

    if ( readdata(argv[1], argv[2]) != 0 ) return(-1);

    // GPU memory allocation variables
    long long *histogramDDGPU, *histogramDRGPU, *histogramRRGPU;
    float *raRealGPU, *declRealGPU, *raSimGPU, *declSimGPU;

    // GPU sizes for memory allocation
    size_t arraybytes = NUM_BINS * sizeof(long long);
    size_t realbytes = NoofReal * sizeof(float);
    size_t simbytes = NoofSim * sizeof(float);

    // Memory allocation on the GPU
    cudaMalloc(&histogramDDGPU, arraybytes);
    cudaMalloc(&histogramDRGPU, arraybytes);
    cudaMalloc(&histogramRRGPU, arraybytes);
    cudaMalloc(&raRealGPU, realbytes);
    cudaMalloc(&declRealGPU, realbytes);
    cudaMalloc(&raSimGPU, simbytes);
    cudaMalloc(&declSimGPU, simbytes);

    // Copy data to the GPU
    cudaMemcpy(raRealGPU, ra_real, realbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(declRealGPU, decl_real, realbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(raSimGPU, ra_sim, simbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(declSimGPU, decl_sim, simbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(histogramDDGPU, histogramDD, arraybytes, cudaMemcpyHostToDevice);
    cudaMemcpy(histogramDRGPU, histogramDR, arraybytes, cudaMemcpyHostToDevice);
    cudaMemcpy(histogramRRGPU, histogramRR, arraybytes, cudaMemcpyHostToDevice);

    // Run the kernels on the GPU
    int blocksPerGrid = (NoofReal + threadsPerBlock - 1) / threadsPerBlock;
    fillBinsOptimizedKernel<<<blocksPerGrid, threadsPerBlock>>>(raRealGPU, declRealGPU, NoofReal, raSimGPU, declSimGPU, NoofSim, histogramDDGPU, histogramDRGPU, histogramRRGPU);

    // Copy the results back to the CPU
    cudaMemcpy(histogramDD, histogramDDGPU, arraybytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(histogramDR, histogramDRGPU, arraybytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(histogramRR, histogramRRGPU, arraybytes, cudaMemcpyDeviceToHost);

    // Free the memory on the GPU
    cudaFree(histogramDDGPU);
    cudaFree(histogramDRGPU);
    cudaFree(histogramRRGPU);
    cudaFree(raRealGPU);
    cudaFree(declRealGPU);
    cudaFree(raSimGPU);
    cudaFree(declSimGPU);

    // Calculate omega values on the CPU
    calculateOmega();

    // Print the results
    printResults();

    kerneltime = 0.0;
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    kerneltime += end-start;
    printf("   Run time = %.lf secs\n", kerneltime);

    free(ra_real);
    free(decl_real);
    free(ra_sim);
    free(decl_sim);

    return(0);
}


int readdata(char *argv1, char *argv2){
  int i,linecount;
  char inbuf[180];
  double ra, dec, phi, theta, dpi;
  FILE *infil;

  printf("   Assuming input data is given in arc minutes!\n");
                          // spherical coordinates phi and theta in radians:
                          // phi   = ra/60.0 * dpi/180.0;
                          // theta = (90.0-dec/60.0)*dpi/180.0;

  dpi = acos(-1.0);
  infil = fopen(argv1,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

  // read the number of galaxies in the input file
  int announcednumber;
  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv1);return(-1);}
  linecount =0;
  while ( fgets(inbuf,180,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv1, linecount);
  else
      {
      printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
      return(-1);
      }

  NoofReal = linecount;
  ra_real   = (float *)calloc(NoofReal,sizeof(float));
  decl_real = (float *)calloc(NoofReal,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i = 0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 )
         {
         printf("   Cannot read line %d in %s\n",i+1,argv1);
         fclose(infil);
         return(-1);
         }
      ra_real[i]   = (float)ra;
      decl_real[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofReal )
      {
      printf("   Cannot read %s correctly\n",argv1);
      return(-1);
      }

  infil = fopen(argv2,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}

  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv2);return(-1);}
  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv2, linecount);
  else
      {
      printf("   %s does not contain %d galaxies but %d\n",argv2, announcednumber,linecount);
      return(-1);
      }

  NoofSim = linecount;
  ra_sim   = (float *)calloc(NoofSim,sizeof(float));
  decl_sim = (float *)calloc(NoofSim,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i =0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 )
         {
         printf("   Cannot read line %d in %s\n",i+1,argv2);
         fclose(infil);
         return(-1);
         }
      ra_sim[i]   = (float)ra;
      decl_sim[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofSim )
      {
      printf("   Cannot read %s correctly\n",argv2);
      return(-1);
      }

  return(0);
}

int getDevice(int deviceNo){
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
       printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                   =   %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels             =   ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if ( device != deviceNo ) printf("   Unable to set device %d, using device %d instead",deviceNo, device);
    else printf("   Using CUDA device %d\n\n", device);

    return(0);
}

