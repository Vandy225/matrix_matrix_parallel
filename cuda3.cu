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
//optimum has been determined to be 16
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


double* unoptimized_mult( float* a, float* b, float* c, int run_number) {



double dt_and_rate[2];

unsigned long long l = dimension;
unsigned long long m = dimension;
unsigned long long n = dimension;

// Load A and B to device memory
float* cuda_A;
//size_t size = l * l * sizeof(float);
unsigned long long size = l*l*sizeof(float);

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
//store the elapsed time
cudaEventElapsedTime(&time_elapsed,start,stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);

//figure out how many operations needed to be done
unsigned long long ops = l * l * ( 2 * l );

time_elapsed = time_elapsed/1000; //change into seconds
//figure out the rate
double rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

  printf ( "\n" );
  printf("Run number: %d\n", run_number);
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed);
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


// Read C from device memory
cudaMemcpy(c, cuda_C, size, cudaMemcpyDeviceToHost);



//return values for main
  dt_and_rate[0]=time_elapsed;
  dt_and_rate[1]=rate;





// Free device memory
cudaFree(cuda_A);
cudaFree(cuda_B);
cudaFree(cuda_C);

return dt_and_rate;
}



//retrieve a matrix element
//device function because this needs to execute on the graphics card, not on host machines
__device__ float retrieve_item(float* a, int row, int col, unsigned long long dim) {
return a[row * dim + col];
}


// Set a matrix element
//device function because this needs to execute on the graphics card, not on host machines
__device__ void set_entry(float* a, int row, int col, float value, unsigned long long dim) {
a[row * dim + col] = value;
}

//get the next sub-block of a matrix when doing blocking multiplication
__device__ float* get_next_block(float* a, int row, int col, unsigned long long dimension) {
float* block_a;
block_a = &a[dimension * BLOCK_SIZE * row + BLOCK_SIZE * col];
return block_a;
}


// Matrix multiplication kernel called by MatMul()
__global__ void blocking_mult_kernel(float* a, float* b, float* c, unsigned long long dimension) {
// Block row and column
int blockRow = blockIdx.y;
int blockCol = blockIdx.x;
// Each thread block computes one sub-matrix of C
float* block_c = get_next_block(c, blockRow, blockCol, dimension);
// Each thread computes one element of the tile of C
// by accumulating results into accumulator
float accumulator = 0.0;
// Thread row and column within the tile of c
int row = threadIdx.y;
int col = threadIdx.x;
// Loop over all A and B's tiles
// Multiply each pair of sub-matrices together
// and accumulate the results
for (int m = 0; m < (dimension / BLOCK_SIZE); ++m) {
// Get sub-matrix of A
float* block_a = get_next_block(a, blockRow, m, dimension);
// Get sub-matrix of B
float* block_b = get_next_block(b, m, blockCol, dimension);
// Shared memory used to store the tiles of A and B
__shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];
// Load tiles from device memory to shared memory
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
// Write the tile of to device memory
// Each thread writes one element
set_entry(block_c, row, col, accumulator, dimension);
}


// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
double* blocking_mult(float* a, float* b, float* c, int run_number) {


double dt_and_rate[2];

unsigned long long l = dimension;
unsigned long long m = dimension;
unsigned long long n = dimension;


// Load A and B to device memory
float* cuda_A;

size_t size = dimension * dimension * sizeof(float);
//allocate device memory
cudaMalloc(&cuda_A, size);
//copy to device
cudaMemcpy(cuda_A, a, size, cudaMemcpyHostToDevice);
//do the same for b
float* cuda_B;
size = dimension * dimension * sizeof(float);
cudaMalloc(&cuda_B, size);
cudaMemcpy(cuda_B, b, size, cudaMemcpyHostToDevice);
// Allocate C in device memory
float* cuda_C;
size = dimension * dimension * sizeof(float);
cudaMalloc(&cuda_C, size);
//set up float for timer
float time_elapsed;


//st up the block and grid for device
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
dim3 dimGrid(dimension / dimBlock.x, dimension / dimBlock.y);
//start timers
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);

//call the CUDA function
blocking_mult_kernel<<<dimGrid, dimBlock>>>(cuda_A, cuda_B, cuda_C, dimension);
//synchronize threads
cudaThreadSynchronize();

cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time_elapsed,start,stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);
//figure out how many operations we have done
unsigned long long ops = l * l * ( 2 * l );

time_elapsed = time_elapsed/1000; //change into seconds
  //get the rate
  double rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

  printf ( "\n" );
  printf("Run number: %d\n", run_number);
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed);
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );



// Read c from device memory
cudaMemcpy(c, cuda_C, size, cudaMemcpyDeviceToHost);

// Free device memory
cudaFree(cuda_A);
cudaFree(cuda_B);
cudaFree(cuda_C);


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

//main function
int main(int argc, char* argv[]){

//used to determine average rates and times
double* temp;
double average_dt=0.0;
double average_rate=0.0;
//get command line options
char options;
    while( (options=getopt(argc,argv,"n:h")) != -1){
        switch(options){
        case 'n':
            dimension = atoi(optarg);
            break;
        case 'h':
            usage(argv);
            exit(0);
        default:
            usage(argv);
            exit(1);
        }
    }
//allocate space
float* a = (float*)malloc(dimension * dimension * sizeof(float));

float* b = (float*)malloc(dimension * dimension * sizeof(float));

float* c = (float*)malloc(dimension * dimension * sizeof(float));

int seed=123456789;

//seed the matrices with random values
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


printf ("Thread Block Size: %d\n", BLOCK_SIZE);

printf( "\n" );
printf( "========================Unoptimized CUDA Multiplication================================\n" );

for(int i =0; i < 10; i++){
temp = unoptimized_mult(a, b, c,i);
average_dt += temp[0];
average_rate += temp[1];

}

average_rate = (double) average_rate/10;
average_dt = (double) average_dt/10;
printf("Average Elapsed Time dT: %f\n", average_dt);
printf("Average Rate: %f\n", average_rate);
average_rate=average_dt=0.0;

printf( "\n" );
printf( "========================Blocking Optimized CUDA Multiplication================================\n" );


for(int i=0; i<10;i++){
temp=blocking_mult(a,b,c,i);
average_dt += temp[0];
average_rate += temp[1];
}

average_rate = (double) average_rate/10;
average_dt = (double) average_dt/10;
printf("Average Elapsed Time dT: %f\n", average_dt);
printf("Average Rate: %f\n", average_rate);
average_rate=average_dt=0.0;



//free host memory
free(a);
free(b);
free(c);

}
