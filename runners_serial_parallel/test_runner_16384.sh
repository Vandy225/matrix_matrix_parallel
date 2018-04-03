#serial runs
/home/vanj4210/parallel_project/matrix_matrix_parallel/serial -n 16384 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/serial_16384.txt


#parallel runs


/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 16384 -t 1 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_1T_16384.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 16384 -t 2 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_2T_16384.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 16384 -t 4 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_4T_16384.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 16384 -t 8 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_8T_16384.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 16384 -t 16 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_16T_16384.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 16384 -t 32 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_32T_16384.txt







