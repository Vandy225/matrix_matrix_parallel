# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <getopt.h>
# include <omp.h>
# include <time.h>

int main ( int argc, char *argv[] );
void r8_mxm ( int l, int m, int n, int num_t );
void unoptimized_serial(int l, int m, int n);
void optimized_serial (int l, int m, int n);
void unoptimized_parallel(int l, int m, int n, int num_t);
void optimized_parallel(int l, int m, int n, int num_t);
void blocking_serial (int l, int m, int n, int block_size);
void blocking_parallel(int l, int m, int n, int block_size, int num_t);
double r8_uniform_01 ( int *seed );
void usage(char* argv[]);


/* Please modify for GPU Experiments */
/* @@@ Shahadat Hossain (SH) March 12, 2018 */
/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:

   <<< SH:  Skeletal c code for performing dense matrix times matrix. >>>
   

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

   @@@ Shahadat Hossain (SH) Nov 15, 2014 

 */
{
  int id;
  int l;
  int m;
  int n;
  //initialize with max number of threads
  int num_t=omp_get_max_threads();

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
    if (num_t > omp_get_max_threads()){
      printf("Invalid number of threads, please choose a value in range 1 - %i\n", omp_get_max_threads());
      exit(0);
    }

  printf ( "\n" );
  printf ( "Dense MXM\n" );
  printf ( "  C/OpenMP version.\n" );
  printf ( "\n" );
  printf ( "  Matrix multiplication tests.\n" );

/*  @@@ SH Note 1a:

   You must read in the dimension of the matrix and the number of threads
   from the command line.
*/
  printf ( "\n" );
  printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );


  //call the matrix-matrix multiplication
  unoptimized_serial( l, m, n);
  optimized_serial(l,m,n);
  unoptimized_parallel(l,m,n,num_t);
  optimized_parallel(l,m,n,num_t);
  //blocking tests, working on 4,8,16,32
  printf( "\n" );
  printf( "========================Blocking================================" );
  blocking_serial (l,m,n,4);
  blocking_serial (l,m,n,8);
  blocking_serial (l,m,n,16);
  blocking_serial (l,m,n,32);
  blocking_serial (l,m,n,64);
  blocking_serial (l,m,n,128);
  blocking_serial (l,m,n,l);

  printf ("\n");
  printf("====================Parallel Blocking=============================");
  blocking_parallel(l,m,n,4,num_t);



/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "Dense MXM:\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;
}



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







void unoptimized_serial ( int l, int m, int n){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  unsigned long long int ops;
  double rate;
  int seed;
  double time_begin;
  double time_elapsed;
  double time_stop;
/*
  Allocate the storage for matrices.
*/
  a = ( double ** ) calloc ( l, sizeof ( double ) );
  b = ( double ** ) calloc ( l , sizeof ( double ) );
  c = ( double ** ) calloc ( m , sizeof ( double ) );
  for ( i = 0; i < l ; i++)
    a[i] = (double *) calloc (n, sizeof (double));
  for ( i = 0; i < l ; i++)
    b[i] = (double *) calloc (m, sizeof (double));
 for ( i = 0; i < m ; i++)
    c[i] = (double *) calloc (n, sizeof (double));
/*
  Assign randomly generated values to the input matrices B and C.
*/
  seed = 123456789;

  for ( k = 0; k < l ; k++ )
    for ( j = 0; j < m; j++)
    {
      b[k][j] = r8_uniform_01 ( &seed );
    }

  for ( k = 0; k < m ; k++ )
    for (j = 0; j < n; j++)
     {
       a[k][j] = r8_uniform_01 ( &seed );
     }


/***************************************************************************************************************/
/* Unoptimized Serial execution section */
time_begin = omp_get_wtime ( );

//naive matrix multiplication
for ( i = 0; i < l; i++)
  {
    for ( j = 0; j < n; j++ )
    {
      c[i][j] = 0.0;
      for ( k = 0; k < m; k++ )
      {
        c[i][j] = c[i][j] + b[i][k] * a[k][j];

      }
    }
  }

time_stop = omp_get_wtime ( );


ops = l * n * ( 2 * m );
time_elapsed = time_stop - time_begin;
rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

printf ( "\n" );
  printf ( "R8_MXM matrix multiplication unoptimized serial timing.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %d\n", l );
  printf ( "  M = %d\n", m );
  printf ( "  N = %d\n", n );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );

  free ( a );
  free ( b );
  free ( c );

  return;

}

void optimized_serial(int l, int m, int n){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  int ops;
  double r;
  double rate;
  int seed;
  double time_begin;
  double time_elapsed;
  double time_stop;
  /************************************************************************************************************/
/* Loop-Optimized Serial execution section */

a = ( double ** ) calloc ( l, sizeof ( double ) );
  b = ( double ** ) calloc ( l , sizeof ( double ) );
  c = ( double ** ) calloc ( m , sizeof ( double ) );
  for ( i = 0; i < l ; i++)
    a[i] = (double *) calloc (n, sizeof (double));
  for ( i = 0; i < l ; i++)
    b[i] = (double *) calloc (m, sizeof (double));
 for ( i = 0; i < m ; i++)
    c[i] = (double *) calloc (n, sizeof (double));
/*
  Assign randomly generated values to the input matrices B and C.
*/
  seed = 123456789;

  for ( k = 0; k < l ; k++ )
    for ( j = 0; j < m; j++)
    {
      b[k][j] = r8_uniform_01 ( &seed );
    }

  for ( k = 0; k < m ; k++ )
    for (j = 0; j < n; j++)
     {
       a[k][j] = r8_uniform_01 ( &seed );
     }

time_begin = omp_get_wtime ( );

//naive matrix multiplication
/*
for ( i = 0; i < l; i++)
  {
    for ( j = 0; j < n; j++ )
    {
      a[i][j] = 0.0;
      for ( k = 0; k < m; k++ )
      {
        a[i][j] = a[i][j] + b[i][k] * c[k][j];

      }
    }
  }*/

for (i=0; i<n; i++){
  for (k=0; k<n; k++) {
    r = a[i][k];
    for (j=0; j<n; j++)
      c[i][j] += r * b[k][j];
  }
}

time_stop = omp_get_wtime ( );

ops = l * n * ( 2 * m );
time_elapsed = time_stop - time_begin;
rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

printf ( "\n" );
  printf ( "Matrix multiplication optimized serial timing.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %d\n", l );
  printf ( "  M = %d\n", m );
  printf ( "  N = %d\n", n );
  printf ( "  Floating point OPS roughly %d\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );

  free ( a );
  free ( b );
  free ( c );

  return;

}
void unoptimized_parallel(int l, int m, int n, int num_t){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  int ops;
  double rate;
  int seed;
  double time_begin;
  double time_elapsed;
  double time_stop;

  a = ( double ** ) calloc ( l, sizeof ( double ) );
  b = ( double ** ) calloc ( l , sizeof ( double ) );
  c = ( double ** ) calloc ( m , sizeof ( double ) );
  for ( i = 0; i < l ; i++)
    a[i] = (double *) calloc (n, sizeof (double));
  for ( i = 0; i < l ; i++)
    b[i] = (double *) calloc (m, sizeof (double));
 for ( i = 0; i < m ; i++)
    c[i] = (double *) calloc (n, sizeof (double));
/*
  Assign randomly generated values to the input matrices B and C.
*/
  seed = 123456789;

  for ( k = 0; k < l ; k++ )
    for ( j = 0; j < m; j++)
    {
      b[k][j] = r8_uniform_01 ( &seed );
    }

  for ( k = 0; k < m ; k++ )
    for (j = 0; j < n; j++)
     {
       a[k][j] = r8_uniform_01 ( &seed );
     }

  time_begin = omp_get_wtime ( );

//set the number of threads
  omp_set_num_threads(num_t);

# pragma omp parallel \
  shared ( a, b, c, l, m, n ) \
  private ( i, j, k )

# pragma omp for
  for ( i = 0; i < l; i++)
  {
    for ( j = 0; j < n; j++ )
    {
      c[i][j] = 0.0;
      for ( k = 0; k < m; k++ )
      {
        c[i][j] = c[i][j] + b[i][k] * a[k][j];

      }
    }
  }
  time_stop = omp_get_wtime ( );

  ops = l * n * ( 2 * m );
  time_elapsed = time_stop - time_begin;
  rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

  printf ( "\n" );
  printf ( "R8_MXM matrix multiplication unoptimized OpenMP timing.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %d\n", l );
  printf ( "  M = %d\n", m );
  printf ( "  N = %d\n", n );
  printf ( "  Floating point OPS roughly %d\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  return;

}

void optimized_parallel(int l, int m, int n, int num_t){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  int ops;
  double rate;
double r;
  int seed;
  double time_begin;
  double time_elapsed;
  double time_stop;

  a = ( double ** ) calloc ( l, sizeof ( double ) );
  b = ( double ** ) calloc ( l , sizeof ( double ) );
  c = ( double ** ) calloc ( m , sizeof ( double ) );
  for ( i = 0; i < l ; i++)
    a[i] = (double *) calloc (n, sizeof (double));
  for ( i = 0; i < l ; i++)
    b[i] = (double *) calloc (m, sizeof (double));
 for ( i = 0; i < m ; i++)
    c[i] = (double *) calloc (n, sizeof (double));
/*
  Assign randomly generated values to the input matrices B and C.
*/
  seed = 123456789;

  for ( k = 0; k < l ; k++ )
    for ( j = 0; j < m; j++)
    {
      b[k][j] = r8_uniform_01 ( &seed );
    }

  for ( k = 0; k < m ; k++ )
    for (j = 0; j < n; j++)
     {
       a[k][j] = r8_uniform_01 ( &seed );
     }

  time_begin = omp_get_wtime ( );

//set the number of threads
  omp_set_num_threads(num_t);

# pragma omp parallel \
  shared ( a, b, c, l, m, n ) \
  private ( i, j, k )

# pragma omp for
  for (i=0; i<n; i++){
    for (k=0; k<n; k++) {
      r = a[i][k];
      for (j=0; j<n; j++)
      c[i][j] += r * b[k][j];
  }
}

  time_stop = omp_get_wtime ( );

  ops = l * n * ( 2 * m );
  time_elapsed = time_stop - time_begin;
  rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

  printf ( "\n" );
  printf ( "Matrix multiplication Loop optimized OpenMP timing.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %d\n", l );
  printf ( "  M = %d\n", m );
  printf ( "  N = %d\n", n );
  printf ( "  Floating point OPS roughly %d\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  return;


}


void blocking_serial (int l, int m, int n, int block_size){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  int ops;
  double rate;
  int seed;
  double time_begin;
  double time_elapsed;
  double time_stop;
/*
  Allocate the storage for matrices.
*/
  a = ( double ** ) calloc ( l, sizeof ( double ) );
  b = ( double ** ) calloc ( l , sizeof ( double ) );
  c = ( double ** ) calloc ( m , sizeof ( double ) );
  for ( i = 0; i < l ; i++)
    a[i] = (double *) calloc (n, sizeof (double));
  for ( i = 0; i < l ; i++)
    b[i] = (double *) calloc (m, sizeof (double));
 for ( i = 0; i < m ; i++)
    c[i] = (double *) calloc (n, sizeof (double));
/*
  Assign randomly generated values to the input matrices B and C.
*/
  seed = 123456789;

  for ( k = 0; k < l ; k++ )
    for ( j = 0; j < m; j++)
    {
      b[k][j] = r8_uniform_01 ( &seed );
    }

  for ( k = 0; k < m ; k++ )
    for (j = 0; j < n; j++)
     {
       a[k][j] = r8_uniform_01 ( &seed );
     }

int en = block_size * (n/block_size);
for (i = 0; i < n; i++)
  for (j = 0; j < n; j++)
    c[i][j] = 0.0;

/***************************************************************************************************************/
/* Unoptimized Serial execution section */
time_begin = omp_get_wtime ( );

/*
//naive matrix multiplication
for ( i = 0; i < l; i++)
  {
    for ( j = 0; j < n; j++ )
    {
      a[i][j] = 0.0;
      for ( k = 0; k < m; k++ )
      {
        a[i][j] = a[i][j] + b[i][k] * c[k][j];

      }
    }
  }
  */
//reproduced with permission from Shahadat, came from webaside code
int kk, jj;
double sum;


for (kk = 0; kk < en; kk += block_size) {
  for (jj = 0; jj < en; jj += block_size) {
    for (i = 0; i < n; i++) {
      for (j = jj; j < jj + block_size; j++) {
        sum = c[i][j];
        for (k = kk; k < kk + block_size; k++) {
          sum += a[i][k]*b[k][j];
        }
        c[i][j] = sum;
      }
    }
  }
}



time_stop = omp_get_wtime ( );


ops = l * n * ( 2 * m );
time_elapsed = time_stop - time_begin;
rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

printf ( "\n" );
  printf ( "R8_MXM matrix multiplication blocking serial timing.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %d\n", l );
  printf ( "  M = %d\n", m );
  printf ( "  N = %d\n", n );
  printf ( "  Block Size = %d\n", block_size );
  printf ( "  Floating point OPS roughly %d\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );

  free ( a );
  free ( b );
  free ( c );

  return;

}

void blocking_parallel(int l, int m, int n, int block_size,int num_t){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  int ops;
  double rate;
  int seed;
  double time_begin;
  double time_elapsed;
  double time_stop;

  a = ( double ** ) calloc ( l, sizeof ( double ) );
  b = ( double ** ) calloc ( l , sizeof ( double ) );
  c = ( double ** ) calloc ( m , sizeof ( double ) );
  for ( i = 0; i < l ; i++)
    a[i] = (double *) calloc (n, sizeof (double));
  for ( i = 0; i < l ; i++)
    b[i] = (double *) calloc (m, sizeof (double));
 for ( i = 0; i < m ; i++)
    c[i] = (double *) calloc (n, sizeof (double));
/*
  Assign randomly generated values to the input matrices B and C.
*/
  seed = 123456789;

  for ( k = 0; k < l ; k++ )
    for ( j = 0; j < m; j++)
    {
      b[k][j] = r8_uniform_01 ( &seed );
    }

  for ( k = 0; k < m ; k++ )
    for (j = 0; j < n; j++)
     {
       a[k][j] = r8_uniform_01 ( &seed );
     }

  omp_set_num_threads(num_t);

  for (i = 0; i < n; i++)
  for (j = 0; j < n; j++)
    c[i][j] = 0.0;

  time_begin = omp_get_wtime ( );

//set the number of threads
  

/*
# pragma omp parallel \
  shared ( a, b, c, l, m, n ) \
  private ( i, j, k )

# pragma omp for
  for ( i = 0; i < l; i++)
  {
    for ( j = 0; j < n; j++ )
    {
      a[i][j] = 0.0;
      for ( k = 0; k < m; k++ )
      {
        a[i][j] = a[i][j] + b[i][k] * c[k][j];

      }
    }
  }*/
  //TO DO: this is breaking, need to rewrite the loop


int kk, jj;
double sum;
int en = block_size * (n/block_size);
printf("here\n");
# pragma omp parallel shared(a,b,c,l,m,n,kk,jj,en) private(i,j,k)

for (kk = 0; kk < en; kk += block_size) {
  for (jj = 0; jj < en; jj += block_size) {
    # pragma omp for
    for (i = 0; i < n; i++) {
      for (j = jj; j < jj + block_size; j++) {
        sum = c[i][j];
        for (k = kk; k < kk + block_size; k++) {
          sum += a[i][k]*b[k][j];
        }
        c[i][j] = sum;
      }
    }
  }
}/*
  for (big_col_num =0; big_col_num < M; big_col_num += 8){
    //iterate through the rows of the big matrix, skipping by block size each time
    for(big_row_num =0; big_row_num < N; big_row_num += 8){
      //start iterating through the small block rows, starting at whatever row we are currently on and 
      //going until we reach the current row + block size
      for(block_row=big_row_num; block_row < big_row_num + 8; block_row++){
        //iterate through the small block columns, starting at whatever column we are currently on and
        //going until we reach the current row + 8
        for(block_col=big_col_num; block_col < big_col_num + 8; block_col++){
          //if we are not on a diagonal element, then need to transpose 
          if(block_row != block_col){
            //transpose the element
            B[block_col][block_row] = A[block_row][block_col];
          }
          //else it is a diagonal element of the big matrix. Store this for later to reduce conflict misses
          //between the small inner matrices and the large outer matrix
          else{
            //hold the diagonal element in a temporary element
            temporary_element = A[block_row][block_col];
            //grab the index of the diagonal element 
            diagonal_index=block_col;
          }
        }
        //this means we have a diagonal element, now we transpose it 
        if (big_row_num == big_col_num){
          B [diagonal_index][diagonal_index] = temporary_element;

        }
      }
    }

  }*/

  time_stop = omp_get_wtime ( );

  ops = l * n * ( 2 * m );
  time_elapsed = time_stop - time_begin;
  rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

  printf ( "\n" );
  printf ( "R8_MXM matrix multiplication blocking OpenMP timing.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %d\n", l );
  printf ( "  M = %d\n", m );
  printf ( "  N = %d\n", n );
  printf ( "  Block Size = %d\n", block_size );
  printf ( "  Floating point OPS roughly %d\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  return;

}





//Blocking multiplication
/*
void bijk(array A, array B, array C, int n, int bsize){
int i, j, k, kk, jj;
double sum;
int en = bsize * (n/bsize);
for (i = 0; i < n; i++)
  for (j = 0; j < n; j++)
    C[i][j] = 0.0;

for (kk = 0; kk < en; kk += bsize) {
  for (jj = 0; jj < en; jj += bsize) {
    for (i = 0; i < n; i++) {
      for (j = jj; j < jj + bsize; j++) {
        sum = C[i][j];
        for (k = kk; k < kk + bsize; k++) {
          sum += A[i][k]*B[k][j];
        }
        C[i][j] = sum;
      }
    }
  }
}
}*/
