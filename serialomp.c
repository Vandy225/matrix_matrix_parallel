# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <getopt.h>
# include <omp.h>
# include <time.h>

int main ( int argc, char *argv[] );
void r8_mxm ( int l, int m, int n, int num_t );
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
  r8_mxm( l, m, n, num_t ); // call the matrix multiplication routine

/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "Dense MXM:\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;
}
/******************************************************************************/

void r8_mxm ( int l, int m, int n, int num_t )

/******************************************************************************/
/*
  Purpose:

    R8_MXM carries out a matrix-matrix multiplication in double precision real  arithmetic.

  Discussion:

    a(lxn) = b(lxm) * c(mxn).

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    Shahadat Hossain (SH) Nov 15, 2014

    Parameters:

    Input: int l, m, n, the dimensions that specify the sizes of the
    a, b, and c matrices.
*/
{
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
       c[k][j] = r8_uniform_01 ( &seed );
     }


/***************************************************************************************************************/
/* Unoptimized Serial execution section */
double serial_begin = omp_get_wtime ( );

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

double serial_end = omp_get_wtime ( );


ops = l * n * ( 2 * m );
double serial_time_elapsed = serial_end - serial_begin;
double serial_rate = ( double ) ( ops ) / serial_time_elapsed / 1000000.0;

printf ( "\n" );
  printf ( "R8_MXM matrix multiplication unoptimized serial timing.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %d\n", l );
  printf ( "  M = %d\n", m );
  printf ( "  N = %d\n", n );
  printf ( "  Floating point OPS roughly %d\n", ops );
  printf ( "  Elapsed time dT = %f\n", serial_time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", serial_rate );

/***********************************************************************************************************/


/************************************************************************************************************/
/* Loop-Optimized Serial execution section */
double serial_begin = omp_get_wtime ( );

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

serial_end = omp_get_wtime ( );

ops = l * n * ( 2 * m );
serial_time_elapsed = serial_end - serial_begin;
serial_rate = ( double ) ( ops ) / serial_time_elapsed / 1000000.0;

printf ( "\n" );
  printf ( "R8_MXM matrix multiplication unoptimized serial timing.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %d\n", l );
  printf ( "  M = %d\n", m );
  printf ( "  N = %d\n", n );
  printf ( "  Floating point OPS roughly %d\n", ops );
  printf ( "  Elapsed time dT = %f\n", serial_time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", serial_rate );

/*************************************************************************************************************/



/*
  Compute A = B * C.
*/
/* 
   @@@ SH Note 2a:
     — The timer function omp_get_wtime() is used to record wallclock time.

     — The parallel directive given in the code below is for information only. 
       Your job is to try and use different directives as well as loop rearrangement 
       and other code optimization that you have learnt in the course to obtain  
       maximum sequential and parallel performance. 
*/ 
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
      a[i][j] = 0.0;
      for ( k = 0; k < m; k++ )
      {
        a[i][j] = a[i][j] + b[i][k] * c[k][j];

      }
    }
  }
  time_stop = omp_get_wtime ( );
/*
  Generate Report.

  @@@ SH Notes 3b :
    In the reporting part, you should also compute and report parallel efficiency.
*/
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
/******************************************************************************/

double r8_uniform_01 ( int *seed )

/******************************************************************************/
/*
  Purpose:

    R8_UNIFORM_01 is a unit pseudorandom double precision real number R8.

  Discussion:

    This routine implements the recursion

      seed = 16807 * seed mod ( 2**31 - 1 )
      unif = seed / ( 2**31 - 1 )

    The integer arithmetic never requires more than 32 bits,
    including a sign bit.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    11 August 2004

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Springer Verlag, pages 201-202, 1983.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, pages 362-376, 1986.

  Parameters:

    Input/output, int *SEED, a seed for the random number generator.

    Output, double R8_UNIFORM_01, a new pseudorandom variate, strictly between
    0 and 1.
*/
{
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