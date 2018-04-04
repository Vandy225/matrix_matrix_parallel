#serial runs
#/home/vanj4210/parallel_project/matrix_matrix_parallel/serial -n 512 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/serial_512.txt


#parallel runs

BIN_PATH=/home/vanj4210/parallel_project/matrix_matrix_parallel/serial
OUTPUT_PATH=/home/vanj4210/parallel_project/matrix_matrix_parallel/output/serial

$BIN_PATH -n 512 > $OUTPUT_PATH/serial_512.out
$BIN_PATH -n 1024 > $OUTPUT_PATH/serial_1024.out
$BIN_PATH -n 2048 > $OUTPUT_PATH/serial_2048.out
$BIN_PATH -n 4096 > $OUTPUT_PATH/serial_4096.out
$BIN_PATH -n 8192 > $OUTPUT_PATH/serial_8192.out
$BIN_PATH -n 16384 > $OUTPUT_PATH/serial_16384.out













