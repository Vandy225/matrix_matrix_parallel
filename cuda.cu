# include <stdlib.h>
# include <iostream>
# include <stdio.h>
# include <math.h>
# include <getopt.h>
# include <omp.h>

#define BLOCK_SIZE 64

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

void usage (char* argv[])
{
  //go through and print out any relevant information for command line arguments
  //to use the program
    printf("Usage: %s -n <matrix dimensions> -t <number of threads>\n", argv[0]);
    printf("Options:\n");
    printf("  -h         Print this help message.\n");
    printf("  -n         Dimensions of the matrices.\n");
    printf("  -t         Number of threads to execute program on\n");
    printf("\nExamples:\n");
    printf("  %s -n 256 -t 8\n", argv[0]);
    printf("  %s -n 4096 -t 16\n", argv[0]);
    //end the program
    exit(0);
}





__global__ void gpu_square_matrix_mult(float *d_a, float *d_b, float *d_result, int n, int block_size) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * block_size + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * block_size + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < block_size; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}








__global__
void unoptimized_serial( int n, float* a, float* b, float* c){
  int i;
  int j;
  int k;

  

  int r;
  //int offset;

  //1D matrix mult

  for (i=0; i < n;i++){
  	for (j=0; j < n; j++){
  		//optimization
  		r=i*n;
  		c[r+j]=0.0;
  		for(k=0; k<n; k++){
  			c[r+j] = c[r+j] + b[r+k] * a[n*k+j];
  		}
  	}
  }

  /*
  //smart 1D mult

  for (i=0; i<n; i++){
  	for (k=0; k<n; k++) {
    r = a[i*n+k];
    offset=i*n;
    for (j=0; j<n; j++)
      c[offset+j] += r * b[k*n+j];
  }
}*/



  return;

}

int main (int argc, char* argv[]){
	unsigned long long int l;
	unsigned long long int m;
	unsigned long long int n;

	unsigned long long int ops;
	double rate;
	float time_elapsed;
  int block_size=BLOCK_SIZE;


	int num_t = 1;


  float *a;
  float *b;
  float *c;
  //int i;
  int j;
  int k;
  
  
  int seed;


  char options;
    while( (options=getopt(argc,argv,"n:t:h")) != -1){
        switch(options){
        case 'n':
            l = m = n = atoi(optarg);
            break;
        case 't':
            num_t = atoi(optarg);
            break;
        case 'h':
            usage(argv);
            exit(0);
        default:
            usage(argv);
            exit(1);
        }
    }




/*
  Allocate the storage for matrices.
*/
  cudaMallocHost((void **) &a, sizeof( float )*l*l);
  cudaMallocHost((void **) &b, sizeof( float )*l*l);
  cudaMallocHost((void **) &c, sizeof( float )*l*l);

  /*
  b = ( float * ) calloc ( l*l, sizeof ( float ) );
  c = ( float * ) calloc ( l*l, sizeof ( float ) );*/
/*
  Assign randomly generated values to the input matrices B and C.
*/
  seed = 123456789;

  for ( k = 0; k < l ; k++ )
    for ( j = 0; j < m; j++)
    {
      b[k*l+j] = r8_uniform_01 ( &seed );
    }

  for ( k = 0; k < m ; k++ )
    for (j = 0; j < n; j++)
     {
       a[k*l+j] = r8_uniform_01 ( &seed );
     }


	
    

  float *a_cuda, *b_cuda, *c_cuda;

  // Allocate Memory on GPU
  cudaMalloc((void **)&a_cuda, l*l*sizeof(float));
  cudaMalloc((void **)&b_cuda, l*l*sizeof(float));
  cudaMalloc((void **)&c_cuda, l*l*sizeof(float));

  //copy over to the GPU

  // Copy inputs to device
  cudaMemcpy(a_cuda, a, l*l*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_cuda, b, l*l*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_cuda, c, l*l*sizeof(float), cudaMemcpyHostToDevice);

  unsigned int grid_rows = (l + block_size - 1) / block_size;
  unsigned int grid_cols = (l + block_size - 1) / block_size;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(block_size, block_size);

  // Run kernel  on the GPU
  // Launch add() kernel on GPU with l blocks, 1 thread per block
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEventRecord(start,0);
  //unoptimized_serial<<<dimGrid,dimBlock>>>(l,a_cuda,b_cuda,c_cuda);
  gpu_square_matrix_mult<<<dimGrid, dimBlock,0>>>(a_cuda, b_cuda, c_cuda, n, block_size); 
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_elapsed,start,stop);

  ops = l * l * ( 2 * l );
  
  rate = ( double ) ( ops ) / time_elapsed*1000 / 1000000.0;

  printf ( "\n" );
  printf ( "CUDA matrix multiplication unoptimized serial.\n" );
  printf ( "Number of threads: %d\n", num_t );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed*1000 );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  cudaFree(a_cuda);
  cudaFree(b_cuda);
  cudaFree(c_cuda);


  cudaFreeHost ( a );
  cudaFreeHost ( b );
  cudaFreeHost ( c );



	return 0;
}



