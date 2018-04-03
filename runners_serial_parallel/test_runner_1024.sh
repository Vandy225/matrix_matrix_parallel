#serial runs
/home/vanj4210/parallel_project/matrix_matrix_parallel/serial -n 1024 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/serial_1024.txt


#parallel runs


/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 1024 -t 1 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_1T_1024.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 1024 -t 2 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_2T_1024.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 1024 -t 4 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_4T_1024.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 1024 -t 8 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_8T_1024.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 1024 -t 16 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_16T_1024.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 1024 -t 32 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_32T_1024.txt







