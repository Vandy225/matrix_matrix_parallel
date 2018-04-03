#serial runs
/home/vanj4210/parallel_project/matrix_matrix_parallel/serial -n 4096 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/serial_4096.txt


#parallel runs


/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 4096 -t 1 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_1T_4096.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 4096 -t 2 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_2T_4096.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 4096 -t 4 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_4T_4096.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 4096 -t 8 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_8T_4096.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 4096 -t 16 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_16T_4096.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 4096 -t 32 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_32T_4096.txt







