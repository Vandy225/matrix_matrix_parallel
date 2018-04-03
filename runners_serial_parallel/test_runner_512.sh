#serial runs
/home/vanj4210/parallel_project/matrix_matrix_parallel/serial -n 512 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/serial_512.txt


#parallel runs


/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 512 -t 1 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_1T_512.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 512 -t 2 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_2T_512.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 512 -t 4 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_4T_512.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 512 -t 8 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_8T_512.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 512 -t 16 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_16T_512.txt

/home/vanj4210/parallel_project/matrix_matrix_parallel/threading -n 512 -t 32 -f /home/vanj4210/parallel_project/matrix_matrix_parallel/output/parallel_32T_512.txt







