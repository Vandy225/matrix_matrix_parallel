#serial runs
#/home/vanj4210/parallel_project/matrix_matrix_parallel/serial -n 512 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/serial_512.txt


#parallel runs

BIN_PATH=/home/vanj4210/parallel_project/matrix_matrix_parallel/threading
OUTPUT_PATH=/home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel




$BIN_PATH -n 2048 -t 32 > $OUTPUT_PATH/parallel_32T_2048.out
$BIN_PATH -n 2048 -t 16 > $OUTPUT_PATH/parallel_16T_2048.out
$BIN_PATH -n 2048 -t 8 > $OUTPUT_PATH/parallel_8T_2048.out
$BIN_PATH -n 2048 -t 4 > $OUTPUT_PATH/parallel_4T_2048.out
$BIN_PATH -n 2048 -t 2 > $OUTPUT_PATH/parallel_2T_2048.out
$BIN_PATH -n 2048 -t 1 > $OUTPUT_PATH/parallel_1T_2048.out













