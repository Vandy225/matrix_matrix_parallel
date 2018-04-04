#!/bin/bash

RUNNER_PATH=/home/vanj4210/parallel_project/matrix_matrix_parallel/runners_parallel


$RUNNER_PATH/parallel_test_runner_512.sh
$RUNNER_PATH/parallel_test_runner_1024.sh
$RUNNER_PATH/parallel_test_runner_2048.sh
$RUNNER_PATH/parallel_test_runner_4096.sh
$RUNNER_PATH/parallel_test_runner_8192.sh
$RUNNER_PATH/parallel_test_runner_16384.sh