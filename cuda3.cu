#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

//global variable for dimensions of matrices
unsigned long long dimension;


//--Function to randomly seed values in the matrices
//provided by Shahadat
double r8_uniform_01 ( int *seed ){
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + 2147483647;
  }

  r = ( double ) ( *seed ) * 4.656612875E-10;

  return r;
}

//We will use this define statement to specify what the size of the thread blocks should be 
#define BLOCK_SIZE 16

__global__ void unoptimized_mult_kernel(float* a, float* b, float* c, unsigned long long dimension) {
// Each thread computes one element of c
// by accumulating results into accumulator
float accumulator = 0.0;
//iterate through the row s and columns of the thread block
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
//if we go out of bounds, stop
if(row > dimension || col > dimension) return;
//since each thread is computing one element of c, grab the appropriate
//element from a's row, and b's column, and multply them and hold them in the accumulator
//then set the value in c
for (int e = 0; e < dimension; ++e)
accumulator += (a[row * dimension + e]) * (b[e * dimension + col]);
c[row * dimension + col] = accumulator;
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE

double* unoptimized_mult( float* a, float* b, float* c, int run_number, char* file_name) {



double dt_and_rate[2];

unsigned long long l = dimension;
unsigned long long m = dimension;
unsigned long long n = dimension;

// Load A and B to device memory
float* cuda_A;
size_t size = l * l * sizeof(float);
cudaMalloc(&cuda_A, size);
cudaMemcpy(cuda_A, a, size, cudaMemcpyHostToDevice);

float* cuda_B;
size = l* l * sizeof(float);
cudaMalloc(&cuda_B, size);
cudaMemcpy(cuda_B, b, size, cudaMemcpyHostToDevice);


// Allocate C in device memory

float* cuda_C;
size = l * l * sizeof(float);
cudaMalloc(&cuda_C, size);



//set up the block and grid dimensions
//using BLOCK_SIZEXBLOCK_SIZE for the blocks
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//setting up the dimensions of the grid
//using this method so that we find the numbers such that the correct number of blocks
//reside within the grid

//e.g. for a 1024* 1024 matrix:

//BLOCK_SIZE = 16
//l = 1024
//the formula is thus:
//(1024+16-1)/16 = 1029/16 = 64.6125 = 64 (because integer division)
//and what happens if we multiply by our thread block size? --> 64 * 16 = 1024, which returns our dimension

//the same reasoning goes for the y dimension
dim3 dimGrid((l + dimBlock.x - 1) / dimBlock.x,(l + dimBlock.y - 1) / dimBlock.y);

//declare variable to find out elapsed time
float time_elapsed;


//CUDA time keeping variables
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);
//run the function on the GPU
unoptimized_mult_kernel<<<dimGrid, dimBlock>>>(cuda_A, cuda_B, cuda_C, dimension);
//synchronize threads
cudaThreadSynchronize();

//CUDA timekeeping stop events
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
//store the elapse dtime
cudaEventElapsedTime(&time_elapsed,start,stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);

//figure out how many operations needed to be done
unsigned long long ops = l * l * ( 2 * l );

time_elapsed = time_elapsed/1000; //change into seconds
//figure out the rate  
double rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

  printf ( "\n" );
  printf ( "CUDA matrix multiplication unoptimized serial.\n" );
  //printf ( "Number of threads: %d\n", num_t );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed);
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


// Read C from device memory
cudaMemcpy(c, cuda_C, size, cudaMemcpyDeviceToHost);


FILE *f = fopen(file_name, "ab");
if (f == NULL)
{
    printf("Error opening file!\n");
    exit(1);
}


fprintf(f, "\n");
//fprintf ( f,"R8_MXM matrix multiplication unoptimized serial timing.\n" );
//fprintf ( f,"  A(LxN) = B(LxM) * C(MxN).\n" );
//fprintf ( f,"  L = %llu\n", l );
//fprintf ( f,"  M = %llu\n", m );
//fprintf ( f,"  N = %llu\n", n );
fprintf(f, "Run number: %d\n", run_number);
fprintf(f,"Floating point OPS roughly %llu\n", (unsigned long long)ops);
fprintf (f,"Elapsed time dT = %f\n", time_elapsed );
fprintf ( f,"Rate = MegaOPS/dT = %f\n", rate );
fclose(f);


  dt_and_rate[0]=time_elapsed;
  dt_and_rate[1]=rate;

  



// Free device memory
cudaFree(cuda_A);
cudaFree(cuda_B);
cudaFree(cuda_C);

return dt_and_rate;
}



// Get a matrix element
__device__ float retrieve_item(float* a, int row, int col, unsigned long long dim) {
return a[row * dim + col];
}


// Set a matrix element
__device__ void set_entry(float* a, int row, int col, float value, unsigned long long dim) {
a[row * dim + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is

// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ float* get_next_block(float* a, int row, int col, unsigned long long dimension) {
float* block_a;
//Asub.width = BLOCK_SIZE;
//Asub.height = BLOCK_SIZE;
//Asub.stride = A.stride;
block_a = &a[dimension * BLOCK_SIZE * row + BLOCK_SIZE * col];
return block_a;
}


// Matrix multiplication kernel called by MatMul()
__global__ void blocking_mult_kernel(float* a, float* b, float* c, unsigned long long dimension) {
// Block row and column
int blockRow = blockIdx.y;
int blockCol = blockIdx.x;
// Each thread block computes one sub-matrix Csub of C
float* block_c = get_next_block(c, blockRow, blockCol, dimension);
// Each thread computes one element of Csub
// by accumulating results into accumulator
float accumulator = 0.0;
// Thread row and column within Csub
int row = threadIdx.y;
int col = threadIdx.x;
// Loop over all the sub-matrices of A and B that are
// required to compute Csub
// Multiply each pair of sub-matrices together
// and accumulate the results
for (int m = 0; m < (dimension / BLOCK_SIZE); ++m) {
// Get sub-matrix Asub of A
float* block_a = get_next_block(a, blockRow, m, dimension);
// Get sub-matrix Bsub of B
float* block_b = get_next_block(b, m, blockCol, dimension);
// Shared memory used to store Asub and Bsub respectively
__shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];
// Load Asub and Bsub from device memory to shared memory
// Each thread loads one element of each sub-matrix
a_shared[row][col] = retrieve_item(block_a, row, col, dimension);
b_shared[row][col] = retrieve_item(block_b, row, col, dimension);
// Synchronize to make sure the sub-matrices are loaded
// before starting the computation
__syncthreads();
// Multiply Asub and Bsub together
for (int e = 0; e < BLOCK_SIZE; ++e)
accumulator += a_shared[row][e] * b_shared[e][col];
// Synchronize to make sure that the preceding
// computation is done before loading two new
// sub-matrices of A and B in the next iteration
__syncthreads();
}
// Write Csub to device memory
// Each thread writes one element
set_entry(block_c, row, col, accumulator, dimension);
}


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
double* blocking_mult(float* a, float* b, float* c, int run_number, char* file_name) {


double dt_and_rate[2];

unsigned long long l = dimension;
unsigned long long m = dimension;
unsigned long long n = dimension;


// Load A and B to device memory

float* cuda_A;

size_t size = dimension * dimension * sizeof(float);
cudaMalloc(&cuda_A, size);

cudaMemcpy(cuda_A, a, size, cudaMemcpyHostToDevice);

float* cuda_B;

size = dimension * dimension * sizeof(float);
cudaMalloc(&cuda_B, size);


cudaMemcpy(cuda_B, b, size, cudaMemcpyHostToDevice);
// Allocate C in device memory
//Matrix d_C;
float* cuda_C;

size = dimension * dimension * sizeof(float);
cudaMalloc(&cuda_C, size);


float time_elapsed;

//struct timeval t1, t2;

//gettimeofday(&t1, 0);
// Invoke kernel
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
dim3 dimGrid(dimension / dimBlock.x, dimension / dimBlock.y);

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);


blocking_mult_kernel<<<dimGrid, dimBlock>>>(cuda_A, cuda_B, cuda_C, dimension);
cudaThreadSynchronize();

cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time_elapsed,start,stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);

unsigned long long ops = l * l * ( 2 * l );

time_elapsed = time_elapsed/1000; //change into seconds
  
  double rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

  printf ( "\n" );
  printf ( "CUDA matrix multiplication shared memory blocking.\n" );
  //printf ( "Number of threads: %d\n", num_t );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed);
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );



// Read C from device memory
cudaMemcpy(c, cuda_C, size, cudaMemcpyDeviceToHost);

// Free device memory
cudaFree(cuda_A);
cudaFree(cuda_B);
cudaFree(cuda_C);

FILE *f = fopen(file_name, "ab");
if (f == NULL)
{
    printf("Error opening file!\n");
    exit(1);
}

fprintf(f, "\n");
//fprintf ( f,"R8_MXM matrix multiplication unoptimized serial timing.\n" );
//fprintf ( f,"  A(LxN) = B(LxM) * C(MxN).\n" );
//fprintf ( f,"  L = %llu\n", l );
//fprintf ( f,"  M = %llu\n", m );
//fprintf ( f,"  N = %llu\n", n );
fprintf(f, "Run number: %d\n", run_number);
fprintf(f,"Floating point OPS roughly %llu\n", (unsigned long long)ops);
fprintf (f,"Elapsed time dT = %f\n", time_elapsed );
fprintf ( f,"Rate = MegaOPS/dT = %f\n", rate );
fclose(f);

dt_and_rate[0]=time_elapsed;
dt_and_rate[1]=rate;

return dt_and_rate;



}

void usage (char* argv[])
{
  //go through and print out any relevant information for command line arguments
  //to use the program
    printf("Usage: %s -n <matrix dimensions>\n", argv[0]);
    printf("Options:\n");
    printf("  -h         Print this help message.\n");
    printf("  -n         Dimensions of the matrices.\n");
    printf("\nExamples:\n");
    printf("  %s -n 256\n", argv[0]);
    printf("  %s -n 4096\n", argv[0]);
    //end the program
    exit(0);
}


int main(int argc, char* argv[]){

char* file_name;
FILE* f;
double* temp;
double average_dt=0.0;
double average_rate=0.0;

char options;
    while( (options=getopt(argc,argv,"n:h:f:")) != -1){
        switch(options){
        case 'n':
            dimension = atoi(optarg);
            break;
        case 'f':
            file_name = (char*) malloc(strlen(optarg));
            strcpy(file_name, optarg);
            break;
        case 'h':
            usage(argv);
            exit(0);
        default:
            usage(argv);
            exit(1);
        }
    }

float* a = (float*)malloc(dimension * dimension * sizeof(float));

float* b = (float*)malloc(dimension * dimension * sizeof(float));

float* c = (float*)malloc(dimension * dimension * sizeof(float));

int seed=123456789;


for(int i = 0; i < dimension; i++){
  for(int j = 0; j < dimension; j++){
    
    a[i*dimension + j] = (float) (r8_uniform_01 ( &seed ));
  }
}
  
for(int i = 0; i < dimension; i++){
  for(int j = 0; j < dimension; j++){
  
    b[i*dimension + j] = (float) (r8_uniform_01 ( &seed ));
  }
}

printf( "\n" );
printf( "========================Unoptimized CUDA Multiplication================================\n" );
f = fopen(file_name, "ab");
fprintf(f,"\n");
fprintf(f,"========================Unoptimized CUDA Multiplication================================\n" );

for(int i =0; i < 10; i++){
temp = unoptimized_mult(a, b, c,i,file_name);
average_dt += temp[0];
average_rate += temp[1];

}

average_rate = (double) average_rate/10;
average_dt = (double) average_dt/10;
f = fopen(file_name, "ab");
fprintf(f, "Average Elapsed Time dT: %f\n", average_dt);
fprintf(f, "Average Rate: %f\n", average_rate);
average_rate=average_dt=0.0;
fclose(f);

printf( "\n" );
printf( "========================Blocking Optimized CUDA Multiplication================================\n" );
f = fopen(file_name, "ab");
fprintf(f,"\n");
fprintf(f,"========================Blocking Optimized CUDA Multiplication================================\n" );

for(int i=0; i<10;i++){
temp=blocking_mult(a,b,c,i,file_name);
average_dt += temp[0];
average_rate += temp[1];
}

average_rate = (double) average_rate/10;
average_dt = (double) average_dt/10;
f = fopen(file_name, "ab");
fprintf(f, "Average Elapsed Time dT: %f\n", average_dt);
fprintf(f, "Average Rate: %f\n", average_rate);
average_rate=average_dt=0.0;
fclose(f);




free(a);
free(b);
free(c);
free(file_name);

}