# include <stdlib.h>
# include <stdio.h>
#include <unistd.h>
#include <string.h>
# include <math.h>
# include <getopt.h>
# include <omp.h>
# include <time.h>

int main ( int argc, char *argv[] );

//parallel unoptimized
double* unoptimized_parallel_static(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t, int run_number);
double* unoptimized_parallel_dynamic(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t, int run_number);
double* unoptimized_parallel_guided(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t, int run_number);
//parallel loop optimized
double* loop_optimized_parallel_static(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t, int run_number);
double* loop_optimized_parallel_dynamic(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t, int run_number);
double* loop_optimized_parallel_guided(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t, int run_number);
//parallel blocking
double* blocking_parallel_static(unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int num_t, int run_number);
double* blocking_parallel_dynamic(unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int num_t, int run_number);
double* blocking_parallel_guided(unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int num_t, int run_number);
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
  double* temp;
  
  unsigned long long int l;
  unsigned long long int m;
  unsigned long long int n;
  double average_dt, average_rate;
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
  printf ( "Matrix Size: %llu\n", l );
  printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", num_t );



  printf( "\n" );
  printf( "========================Unoptimized Parallel Static================================\n" );
  for(i =0; i < 10; i ++){
    
    temp= unoptimized_parallel_static(l,m,n,num_t,i);
    average_dt += temp[0];
    average_rate += temp[1];
  }
  average_rate = (double) average_rate/10;
  average_dt = (double) average_dt/10;
  printf("Average Elapsed Time dT: %f\n", average_dt);
  printf("Average Rate: %f\n", average_rate);
  average_rate=average_dt=0.0;

  printf( "\n" );
  printf( "========================Unoptimized Parallel Static End================================\n" );




  printf( "\n" );
  printf( "========================Unoptimized Parallel Dynamic================================\n" );
  for(i =0; i < 10; i++){
    temp=unoptimized_parallel_dynamic(l,m,n,num_t,i);
    average_dt += temp[0];
    average_rate += temp[1];

  }
  average_rate = (double) average_rate/10;
  average_dt = (double) average_dt/10;
  printf("Average Elapsed Time dT: %f\n", average_dt);
  printf("Average Rate: %f\n", average_rate);
  average_rate=average_dt=0.0;
  printf( "\n" );
  printf( "========================Unoptimized Parallel Dynamic End================================\n" );



  printf( "\n" );
  printf( "========================Unoptimized Parallel Guided================================\n" );
  for(i =0; i< 10; i++){
    temp = unoptimized_parallel_guided(l,m,n,num_t, i);
    average_dt += temp[0];
    average_rate += temp[1];

  }
  average_rate = (double) average_rate/10;
  average_dt = (double) average_dt/10;
  printf("Average Elapsed Time dT: %f\n", average_dt);
  printf("Average Rate: %f\n", average_rate);
  average_rate=average_dt=0.0;
  printf( "\n" );
  printf( "========================Unoptimized Parallel Guided End================================\n" );





  printf( "\n" );
  printf( "========================Loop Optimized Parallel Static================================\n" );
  for (i =0; i < 10; i++){
    temp=loop_optimized_parallel_static(l,m,n,num_t,i);
    average_dt += temp[0];
    average_rate += temp[1];
  }
  average_rate = (double) average_rate/10;
  average_dt = (double) average_dt/10;
  printf("Average Elapsed Time dT: %f\n", average_dt);
  printf("Average Rate: %f\n", average_rate);
  average_rate=average_dt=0.0;
  printf( "\n" );
  printf( "========================Loop Optimized Parallel Static End================================\n" );



  printf( "\n" );
  printf( "========================Loop Optimized Parallel Dynamic================================\n" );
  for(i=0;i<10;i++){
    temp=loop_optimized_parallel_dynamic(l,m,n,num_t,i);
    average_dt += temp[0];
    average_rate += temp[1];
  }
  average_rate = (double) average_rate/10;
  average_dt = (double) average_dt/10;
  printf("Average Elapsed Time dT: %f\n", average_dt);
  printf("Average Rate: %f\n", average_rate);
  average_rate=average_dt=0.0;
  printf( "\n" );
  printf( "========================Loop Optimized Parallel Dynamic End================================\n" );
  

  printf( "\n" );
  printf( "========================Loop Optimized Parallel Guided================================\n" );  
  for(i=0; i<10;i++){
    temp=loop_optimized_parallel_guided(l,m,n,num_t,i);
    average_dt += temp[0];
    average_rate += temp[1];
  }
  average_rate = (double) average_rate/10;
  average_dt = (double) average_dt/10;
  printf("Average Elapsed Time dT: %f\n", average_dt);
  printf("Average Rate: %f\n", average_rate);
  average_rate=average_dt=0.0;
  printf( "\n" );
  printf( "========================Loop Optimized Parallel Guided End================================\n" );
 

  int j=0;

  
  

  printf ("\n");
  printf("====================Parallel Blocking Static=============================\n");
  for(j=0; j<10;j++){
      temp=blocking_parallel_static(l,m,n,16,num_t,j);
      average_dt += temp[0];
      average_rate += temp[1];

    }
    average_rate = (double) average_rate/10;
    average_dt = (double) average_dt/10;
    printf("Average Elapsed Time dT: %f\n", average_dt);
    printf("Average Rate: %f\n", average_rate);
    average_rate=average_dt=0.0;
    printf ("\n");
    printf("====================Parallel Blocking Static End=============================\n");
    
  
  

  printf ("\n");
  printf("====================Parallel Blocking Dynamic=============================\n");
    for(j=0;j<10;j++){
      temp=blocking_parallel_dynamic(l,m,n,16,num_t,j);
      average_dt += temp[0];
      average_rate += temp[1];
    }
    average_rate = (double) average_rate/10;
    average_dt = (double) average_dt/10;
    printf("Average Elapsed Time dT: %f\n", average_dt);
    printf("Average Rate: %f\n", average_rate);
    average_rate=average_dt=0.0;

  printf ("\n");
  printf("====================Parallel Blocking Dynamic End=============================\n");
    
  
  

  printf ("\n");
  printf("====================Parallel Blocking Guided=============================\n");
    for(j=0; j<10;j++){
      temp=blocking_parallel_guided(l,m,n,16,num_t,j);
      average_dt += temp[0];
      average_rate += temp[1];
    }
    average_rate = (double) average_rate/10;
    average_dt = (double) average_dt/10;
    printf("Average Elapsed Time dT: %f\n", average_dt);
    printf("Average Rate: %f\n", average_rate);
    average_rate=average_dt=0.0;

  printf ("\n");
  printf("====================Parallel Blocking Guided End=============================\n");


    
 
  



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
    printf("  -f         Name of output file.\n");
    printf("\nExamples:\n");
    printf("  %s -n 256 -t 8 > output.txt\n", argv[0]);
    printf("  %s -n 4096 -t 16 > parallel.txt\n", argv[0]);
    //end the program
    exit(0);
}










double* unoptimized_parallel_static(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t, int run_number){
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

double* unoptimized_parallel_dynamic(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t, int run_number){
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

double* unoptimized_parallel_guided(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t, int run_number){
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

double* loop_optimized_parallel_static(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t, int run_number){
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

double* loop_optimized_parallel_dynamic(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t, int run_number){
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
double* loop_optimized_parallel_guided(unsigned long long int l, unsigned long long int m, unsigned long long int n, int num_t, int run_number){
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





double* blocking_parallel_static(unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int num_t,int run_number){
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

double* blocking_parallel_dynamic(unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int num_t, int run_number){
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

double* blocking_parallel_guided(unsigned long long int l, unsigned long long int m, unsigned long long int n, int block_size, int num_t, int run_number){
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





