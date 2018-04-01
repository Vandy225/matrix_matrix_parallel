# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <getopt.h>
# include <omp.h>
# include <time.h>

int main ( int argc, char *argv[] );
//serial
void unoptimized_serial(unsigned long long int l, unsigned long long int m, unsigned long long int n);
void loop_optimized_serial (unsigned long long int l, unsigned long long int m, unsigned long long int n);
void blocking_serial (unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size);
//parallel unoptimized
void unoptimized_parallel_static(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t);
void unoptimized_parallel_dynamic(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t);
void unoptimized_parallel_guided(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t);
//parallel loop optimized
void loop_optimized_parallel_static(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t);
void loop_optimized_parallel_dynamic(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t);
void loop_optimized_parallel_guided(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t);
//parallel blocking
void blocking_parallel_static(unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int num_t);
void blocking_parallel_dynamic(unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int num_t);
void blocking_parallel_guided(unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int num_t);
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
  //int id;
  int i;
  unsigned long long int l;
  unsigned long long int m;
  unsigned long long int n;
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

  loop_optimized_serial(l,m,n);
  unoptimized_parallel_static(l,m,n,num_t);
  unoptimized_parallel_dynamic(l,m,n,num_t);
  unoptimized_parallel_guided(l,m,n,num_t);
  loop_optimized_parallel_static(l,m,n,num_t);
  loop_optimized_parallel_dynamic(l,m,n,num_t);
  loop_optimized_parallel_guided(l,m,n,num_t);
  //blocking tests, working on 4,8,16,32
  printf( "\n" );
  printf( "========================Serial Blocking================================" );
  for (i=2; i < n; i*= 2){
    blocking_serial (l,m,n,i);
  }
  

  printf ("\n");
  printf("====================Parallel Blocking Static=============================");
  for (i=2; i < n; i*= 2){
    blocking_parallel_static(l,m,n,i,num_t);
  }
  

  printf ("\n");
  printf("====================Parallel Blocking Dynamic=============================");
  for (i=2; i < n; i*= 2){
    blocking_parallel_dynamic(l,m,n,i,num_t);
  }
  

  printf ("\n");
  printf("====================Parallel Blocking Guided=============================");
  for (i=2; i < n; i*= 2){
    blocking_parallel_guided(l,m,n,i,num_t);
  }
  



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







void unoptimized_serial ( unsigned long long int l, unsigned long long int m, unsigned long long int n){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  unsigned long long int ops;
  long double rate;
  int seed;
  double time_begin;
  double time_elapsed;
  double time_stop;

  unsigned long long test = l * n * ( 2 * m );

  printf("Test %llu\n", test);
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
rate = ( long double ) ( ops ) / time_elapsed / 1000000.0;

printf ( "\n" );
  printf ( "R8_MXM matrix multiplication unoptimized serial timing.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  //printf ( "  Floating point OPS roughly %llu\n", ops );
  //printf("Floating point OPS roughly %llu\n", (unsigned long long)(ops & 0xFFFFFFFFFFFFFFFF));
  printf("Floating point OPS roughly %llu\n", (unsigned long long)ops);
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %Lf\n", rate );



  

FILE *f = fopen("./output/unoptimized_serial.txt", "w+");
if (f == NULL)
{
    printf("Error opening file!\n");
    exit(1);
}

fprintf(f, "\n");
fprintf ( f,"R8_MXM matrix multiplication unoptimized serial timing.\n" );
fprintf ( f,"  A(LxN) = B(LxM) * C(MxN).\n" );
fprintf ( f,"  L = %llu\n", l );
fprintf ( f,"  M = %llu\n", m );
fprintf ( f,"  N = %llu\n", n );
fprintf(f,"Floating point OPS roughly %llu\n", (unsigned long long)ops);
fprintf (f,"  Elapsed time dT = %f\n", time_elapsed );
fprintf ( f,"  Rate = MegaOPS/dT = %Lf\n", rate );
fclose(f);



  free ( a );
  free ( b );
  free ( c );

  return;

}


void loop_optimized_serial(unsigned long long int l, unsigned long long int m, unsigned long long int n){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  unsigned long long ops;
  double r;
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
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );

  free ( a );
  free ( b );
  free ( c );

  return;

}
void unoptimized_parallel_static(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  unsigned long long ops;
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


  for ( i = 0; i < l; i++)
  {
    # pragma omp for
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

  ops = (unsigned long long)l * n * ( 2 * m );
  time_elapsed = time_stop - time_begin;
  rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

  printf ( "\n" );
  printf ( "R8_MXM matrix multiplication unoptimized OpenMP timing, static schedule.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  return;

}

void unoptimized_parallel_dynamic(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  unsigned long long ops;
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


  for ( i = 0; i < l; i++)
  {
    # pragma omp for schedule(dynamic)
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

  ops = (unsigned long long)l * n * ( 2 * m );
  time_elapsed = time_stop - time_begin;
  rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

  printf ( "\n" );
  printf ( "R8_MXM matrix multiplication unoptimized OpenMP timing, dynamic schedule.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  return;

}

void unoptimized_parallel_guided(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  unsigned long long ops;
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


  for ( i = 0; i < l; i++)
  {
    # pragma omp for schedule(guided)
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

  ops = (unsigned long long)l * n * ( 2 * m );
  time_elapsed = time_stop - time_begin;
  rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

  printf ( "\n" );
  printf ( "R8_MXM matrix multiplication unoptimized OpenMP timing, guided schedule.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  return;

}

void loop_optimized_parallel_static(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  unsigned long long int ops;
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


  for (i=0; i<n; i++){
    # pragma omp for
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
  printf ( "Matrix multiplication Loop optimized OpenMP timing, static schedule.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  return;


}

void loop_optimized_parallel_dynamic(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  unsigned long long int ops;
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


  for (i=0; i<n; i++){
    # pragma omp for schedule(dynamic)
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
  printf ( "Matrix multiplication Loop optimized OpenMP timing, dynamic schedule.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  return;


}
void loop_optimized_parallel_guided(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t){
  double **a;
  double **b;
  double **c;
  int i;
  int j;
  int k;
  unsigned long long int ops;
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


  for (i=0; i<n; i++){
    # pragma omp for schedule(guided)
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
  printf ( "Matrix multiplication Loop optimized OpenMP timing, guided schedule.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  return;


}





void blocking_serial (unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size){
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

  a = ( double ** ) calloc ( l, sizeof ( double ) );
  b = ( double ** ) calloc ( l , sizeof ( double ) );
  c = ( double ** ) calloc ( m , sizeof ( double ) );
  for ( i = 0; i < l ; i++)
    a[i] = (double *) calloc (n, sizeof (double));
  for ( i = 0; i < l ; i++)
    b[i] = (double *) calloc (m, sizeof (double));
 for ( i = 0; i < m ; i++)
    c[i] = (double *) calloc (n, sizeof (double));

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


time_begin = omp_get_wtime ( );


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
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Block Size = %d\n", block_size );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );

  free ( a );
  free ( b );
  free ( c );

  return;

}

void blocking_parallel_static(unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int num_t){
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

  a = ( double ** ) calloc ( l, sizeof ( double ) );
  b = ( double ** ) calloc ( l , sizeof ( double ) );
  c = ( double ** ) calloc ( m , sizeof ( double ) );
  for ( i = 0; i < l ; i++)
    a[i] = (double *) calloc (n, sizeof (double));
  for ( i = 0; i < l ; i++)
    b[i] = (double *) calloc (m, sizeof (double));
 for ( i = 0; i < m ; i++)
    c[i] = (double *) calloc (n, sizeof (double));

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


int kk, jj;
double sum;
int en = block_size * (n/block_size);
# pragma omp parallel shared(a,b,c,l,m,n,en) private(i,j,k,jj,kk,sum)

//in the block's rows
for (kk = 0; kk < en; kk += block_size) {
  //in the block's columns
  for (jj = 0; jj < en; jj += block_size) {
    //parallelize this
    # pragma omp for
    for (i = 0; i < n; i++) {
      for (j = jj; j < jj + block_size; j++) {
        sum = c[i][j];
        for (k = kk; k < kk + block_size; k++) {
          sum += a[i][k]*b[k][j];
        }
        #pragma omp critical
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
  printf ( "R8_MXM matrix multiplication blocking OpenMP timing, static schedule.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Block Size = %d\n", block_size );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  return;

}

void blocking_parallel_dynamic(unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int num_t){
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

  a = ( double ** ) calloc ( l, sizeof ( double ) );
  b = ( double ** ) calloc ( l , sizeof ( double ) );
  c = ( double ** ) calloc ( m , sizeof ( double ) );
  for ( i = 0; i < l ; i++)
    a[i] = (double *) calloc (n, sizeof (double));
  for ( i = 0; i < l ; i++)
    b[i] = (double *) calloc (m, sizeof (double));
 for ( i = 0; i < m ; i++)
    c[i] = (double *) calloc (n, sizeof (double));

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


int kk, jj;
double sum;
int en = block_size * (n/block_size);
# pragma omp parallel shared(a,b,c,l,m,n,en) private(i,j,k,jj,kk,sum)

//in the block's rows
for (kk = 0; kk < en; kk += block_size) {
  //in the block's columns
  for (jj = 0; jj < en; jj += block_size) {
    //parallelize this
    # pragma omp for schedule(dynamic)
    for (i = 0; i < n; i++) {
      for (j = jj; j < jj + block_size; j++) {
        sum = c[i][j];
        for (k = kk; k < kk + block_size; k++) {
          sum += a[i][k]*b[k][j];
        }
        #pragma omp critical
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
  printf ( "R8_MXM matrix multiplication blocking OpenMP timing, dynamic schedule.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Block Size = %d\n", block_size );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  return;

}

void blocking_parallel_guided(unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int num_t){
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

  a = ( double ** ) calloc ( l, sizeof ( double ) );
  b = ( double ** ) calloc ( l , sizeof ( double ) );
  c = ( double ** ) calloc ( m , sizeof ( double ) );
  for ( i = 0; i < l ; i++)
    a[i] = (double *) calloc (n, sizeof (double));
  for ( i = 0; i < l ; i++)
    b[i] = (double *) calloc (m, sizeof (double));
 for ( i = 0; i < m ; i++)
    c[i] = (double *) calloc (n, sizeof (double));

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


int kk, jj;
double sum;
int en = block_size * (n/block_size);
# pragma omp parallel shared(a,b,c,l,m,n,en) private(i,j,k,jj,kk,sum)

//in the block's rows
for (kk = 0; kk < en; kk += block_size) {
  //in the block's columns
  for (jj = 0; jj < en; jj += block_size) {
    //parallelize this
    # pragma omp for schedule(guided)
    for (i = 0; i < n; i++) {
      for (j = jj; j < jj + block_size; j++) {
        sum = c[i][j];
        for (k = kk; k < kk + block_size; k++) {
          sum += a[i][k]*b[k][j];
        }
        #pragma omp critical
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
  printf ( "R8_MXM matrix multiplication blocking OpenMP timing, guided schedule.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %llu\n", l );
  printf ( "  M = %llu\n", m );
  printf ( "  N = %llu\n", n );
  printf ( "  Block Size = %d\n", block_size );
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  return;

}





