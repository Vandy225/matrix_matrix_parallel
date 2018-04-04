#!/bin/bash

RUNNER_PATH=/home/vanj4210/parallel_project/matrix_matrix_parallel/runners_cuda/

$RUNNER_PATH/test_runner_cuda_512.sh
$RUNNER_PATH/test_runner_cuda_1024.sh
$RUNNER_PATH/test_runner_cuda_2048.sh  
$RUNNER_PATH/test_runner_cuda_2048.sh
$RUNNER_PATH/test_runner_cuda_4096.sh
$RUNNER_PATH/test_runner_cuda_8192.sh
$RUNNER_PATH/test_runner_cuda_16384.sh
  
