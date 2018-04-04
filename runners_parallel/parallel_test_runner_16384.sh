#serial runs
#/home/vanj4210/parallel_project/matrix_matrix_parallel/serial -n 16384 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/serial_16384.txt


#parallel runs

BIN_PATH=/home/vanj4210/parallel_project/matrix_matrix_parallel/threading
OUTPUT_PATH=/home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel


$BIN_PATH -n 16384 -t 32 > $OUTPUT_PATH/parallel_32T_16384.out
$BIN_PATH -n 16384 -t 16 > $OUTPUT_PATH/parallel_16T_16384.out
$BIN_PATH -n 16384 -t 8 > $OUTPUT_PATH/parallel_8T_16384.out
$BIN_PATH -n 16384 -t 4 > $OUTPUT_PATH/parallel_4T_16384.out
$BIN_PATH -n 16384 -t 2 > $OUTPUT_PATH/parallel_2T_16384.out
$BIN_PATH -n 16834 -t 1 > $OUTPUT_PATH/parallel_1T_16384.out













