#serial runs
/home/vanj4210/parallel_project/matrix_matrix_parallel/serial -n 8192 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/serial_8192.txt


#parallel runs


/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 8192 -t 1 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_1T_8192.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 8192 -t 2 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_2T_8192.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 8192 -t 4 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_4T_8192.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 8192 -t 8 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_8T_8192.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 8192 -t 16 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_16T_8192.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 8192 -t 32 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_32T_8192.txt







