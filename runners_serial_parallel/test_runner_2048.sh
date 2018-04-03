#serial runs
/home/vanj4210/parallel_project/matrix_matrix_parallel/serial -n 2048 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/serial_2048.txt


#parallel runs


/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 2048 -t 1 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_1T_2048.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 2048 -t 2 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_2T_2048.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 2048 -t 4 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_4T_2048.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 2048 -t 8 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_8T_2048.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 2048 -t 16 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_16T_2048.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 2048 -t 32 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_32T_2048.txt







