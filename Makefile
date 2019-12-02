CC=nvcc

gpuhello:gpuhello.cu
	$(CC) -o gpuhello gpuhello.cu

