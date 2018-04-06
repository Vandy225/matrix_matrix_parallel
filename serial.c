# include <stdlib.h>
# include <stdio.h>
#include <unistd.h>
#include <string.h>
# include <math.h>
# include <getopt.h>
# include <omp.h>
# include <time.h>

#define BLOCK_SIZE 16


//serial
double* unoptimized_serial(unsigned long long int l, unsigned long long int m, unsigned long long int n, int run_number);
double* loop_optimized_serial (unsigned long long int l, unsigned long long int m, unsigned long long int n, int run_number);
double* blocking_serial (unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int run_number);



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
  
    printf("Usage: %s -n <matrix dimensions> -t <number of threads>\n", argv[0]);
    printf("Options:\n");
    printf("  -h         Print this help message.\n");
    printf("  -n         Dimensions of the matrices.\n");
    printf("\nExamples:\n");
    printf("  %s -n 256 > output.txt\n", argv[0]);
    printf("  %s -n 4096 > output.txt\n", argv[0]);
    //end the program
    exit(0);
}


int main ( int argc, char *argv[] )


{
  int i;
  double* temp;
  unsigned long long int l;
  unsigned long long int m;
  unsigned long long int n;
  double average_dt, average_rate;

  char options;
    while( (options=getopt(argc,argv,"n:h")) != -1){
        switch(options){
        case 'n':
            l = m = n = atoi(optarg);
            break;
        case 'h':
            usage(argv);
            exit(0);
        default:
            usage(argv);
            exit(1);
        }
    }


  printf ( "\n" );
  printf ( "Serial Processing\n");



  printf ( "Matrix Size: %llu\n", l);




  
  printf( "\n" );
  printf( "========================Unoptimized Serial================================\n" );
  for (i =0; i < 10; i++ ){
    temp = unoptimized_serial( l, m, n, i);
    average_dt += temp[0];
    average_rate += temp[1];

  }
  average_rate = (double) average_rate/10;
  average_dt = (double) average_dt/10;
  
  printf("Average Elapsed Time dT: %f\n", average_dt);
  printf("Average Rate: %f\n", average_rate);
  average_rate=average_dt=0.0;

  printf( "\n" );
  printf( "========================Unoptimized Serial End================================\n" );






  printf( "\n" );
  printf( "========================Loop Optimized Serial================================\n" );
  for (i =0; i < 10; i++){
    temp = loop_optimized_serial(l,m,n,i);
    average_dt += temp[0];
    average_rate += temp[1];

  }
  average_rate = (double) average_rate/10;
  average_dt = (double) average_dt/10;
  printf("Average Elapsed Time dT: %f\n", average_dt);
  printf("Average Rate: %f\n", average_rate);
  average_rate=average_dt=0.0;

  printf( "\n" );
  printf( "========================Loop Optimized Serial End================================\n" );

  int j=0;


  printf( "\n" );
  printf( "========================Serial Blocking================================" );
 
    for(j=0; j < 10; j++){
      temp = blocking_serial (l,m,n,BLOCK_SIZE,j);
      average_dt += temp[0];
      average_rate += temp[1];
    }
    average_rate = (double) average_rate/10;
    average_dt = (double) average_dt/10;
 
    printf("Average Elapsed Time dT: %f\n", average_dt);
    printf("Average Rate: %f\n", average_rate);
    average_rate=average_dt=0.0;

 
  printf( "\n" );
  printf( "========================Serial Blocking End================================" );

  return 0;
}

  double* unoptimized_serial (unsigned long long int l, unsigned long long int m, unsigned long long int n, int run_number){
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

  double dt_and_rate[2];

  unsigned long long test = l * n * ( 2 * m );

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
  printf("Run number: %d\n", run_number);
  printf("Floating point OPS roughly %llu\n", (unsigned long long)ops);
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  dt_and_rate[0] = time_elapsed;
  dt_and_rate[1] = rate;

  return dt_and_rate;

}


double* loop_optimized_serial(unsigned long long int l, unsigned long long int m, unsigned long long int n, int run_number){
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
  double dt_and_rate[2];
  

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
  printf("Run number: %d\n", run_number);
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );

  free ( a );
  free ( b );
  free ( c );

  dt_and_rate[0]=time_elapsed;
  dt_and_rate[1]=rate;

  return dt_and_rate;

}

double* blocking_serial (unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int run_number){
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
  double dt_and_rate[2];

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
  
  printf ( "  Block Size = %d\n", block_size );
  printf("Run number: %d\n", run_number);
  printf ( "  Floating point OPS roughly %llu\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );


  free ( a );
  free ( b );
  free ( c );

  dt_and_rate[0]=time_elapsed;
  dt_and_rate[1]=rate;

  return dt_and_rate;

}