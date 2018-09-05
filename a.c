#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
void matrix_init(int r, int c, float matrix[r][c]);
void print_non_block(int r, int c, float matrix[r][c]);
int result_index(int index1, int index2, int N);
float row_product(int c, float row1[c], float row2[c]);

int main(int argc, char **argv) {
	int myid;
    int numprocs;
	int N = atoi(argv[1]);
	int M = atoi(argv[2]);
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Status status;
	if(N % numprocs != 0){
		printf("N must be divisible by the number of processor\n");
		MPI_Finalize();
		return 0;
	}
	if(N % 2 == 0 || numprocs % 2 == 0){
		printf("N and the number of processor must be odd number\n");
		MPI_Finalize();
		return 0;
	}
	//after row division, each processor would get row_amount rows
	int row_amount = N/numprocs;

	int i,j;
	int i1 = 0;

	if(numprocs == 1){
		float matrix[N][M];
		matrix_init(N,M,matrix);
		print_non_block(N,M,matrix);
	}

	if(numprocs > 1 && myid == 0){
		MPI_Request request;
		//create matrix
		float matrix[N][M];
		matrix_init(N,M,matrix);
		float my_matrix[row_amount][M];
		//for final result, n-1, n-2, .... 1, therefore the total amount element in final result is (N-1)*(1+N-1)/2
		float *final = (float*)calloc((N-1)*(1+N-1)/2, sizeof(float));
		int p_count;
		int current_row = 0;
		//divide matrix, send sub-matrix to every processor
		for(p_count = 0; p_count < numprocs; p_count++){
			int sending_count = 0;
			//make matrix for processor 0 itself
			if(p_count == 0){
				for(i = 0; i < row_amount; i++){
					for(j = 0; j < M; j++){
						my_matrix[i][j] = matrix[current_row][j];
					}
					current_row++;
				}
			}else{
				//send sub-matrix
				float sending_1d_matrix[row_amount * M];
				for(i = 0; i < row_amount; i++){
					for(j = 0; j < M; j++){
						sending_1d_matrix[sending_count] = matrix[current_row][j];
						sending_count++;
					}
					current_row++;
				}
				// MPI_Request send_sub_matrix_request;
				MPI_Isend(sending_1d_matrix, row_amount * M, MPI_FLOAT, p_count, 1, MPI_COMM_WORLD, &request);
			}
		}

		if(row_amount != 1){
			//do first calculation for itself
			for(i = 0; i < row_amount-1; i++){
				for(i1 = i+1; i1 < row_amount; i1++ ){
					int idx = result_index(i,i1,N);
					final[idx] = row_product(M,my_matrix[i], my_matrix[i1]);
				}
			}
			//each processor get row_amount rows, ((row_amount-1)+(row_amount-2)+ ... 1)
			int result_amount_each_p = (row_amount-1)*(1+row_amount-1)/2;
			int *all_idx = (int*)calloc((numprocs - 1)*result_amount_each_p, sizeof(int));
			float *all_value = (float*)calloc((numprocs - 1)*result_amount_each_p, sizeof(float));

			MPI_Request reqs1;
			MPI_Request reqs2;
			MPI_Status status1;
			MPI_Status status2;
			//receive all first calculation result
			for(p_count = 1; p_count < numprocs; p_count++){
				MPI_Irecv((all_idx+(p_count-1)*result_amount_each_p), result_amount_each_p, MPI_INT, p_count, 1, MPI_COMM_WORLD, &reqs1);
				MPI_Irecv((all_value+(p_count-1)*result_amount_each_p), result_amount_each_p, MPI_FLOAT, p_count, 2, MPI_COMM_WORLD, &reqs2);

				MPI_Wait ( &reqs1, &status1);
				MPI_Wait ( &reqs2, &status2);
			}
			//for each first calculation result, set it to corresponding position in final result
			for(i = 0; i < (numprocs - 1)*result_amount_each_p; i++){
				final[all_idx[i]] = all_value[i];
			}
			free(all_idx);
			free(all_value);
		}

		//iteration, communication stuff
		//set arrays
		//for sending data
		float *communication_send_1d_array=(float*)malloc(sizeof(float)*row_amount * M);
		int *communication_send_row_idx=(int*)malloc(sizeof(int)*row_amount);
		//for receiving data
		float *communication_receive_1d_array=(float*)malloc(sizeof(float)*row_amount * M);
		int *communication_receive_row_idx=(int*)malloc(sizeof(int)*row_amount);
		//for changing receive 1d array to 2d array for calculation
		float **communication_receive_2d_array = (float **)malloc(row_amount * sizeof(float *));
		for (i = 0; i < row_amount; i++) {
			communication_receive_2d_array[i] = (float *)malloc(M * sizeof(float));
		}

		//
		int sending_count = 0;
		i = 0;
		j = 0;
		//initialize data for sending 1d array(the data processor 0 have)
		for(sending_count = 0; sending_count < row_amount * M; sending_count++){
			communication_send_1d_array[sending_count] = my_matrix[i][j];
			j++;
			if(j == M){
				j = 0;
				i++;
			}
		}
		//initialize data for sending index(just 0,1,2...row_amount-1)
		sending_count = 0;
		for(sending_count = 0; sending_count < row_amount; sending_count++){
			communication_send_row_idx[sending_count] = sending_count;
		}
		int iter = 0;
		//dest is 2
		int dest;
		dest = myid+1;
		//source is the last processor, i.e. numprocs-1
		int src;
		src = numprocs - 1;
		int *part_final_idx;
		float *part_final_value;
		for(iter = 0; iter < (numprocs-1)/2 ; iter++){
			MPI_Request reqs1;
			MPI_Request reqs2;
			MPI_Status status1;
			MPI_Status status2;
			//send index and rows
			MPI_Isend(communication_send_row_idx, row_amount, MPI_INT, dest, 1, MPI_COMM_WORLD, &reqs1);
			MPI_Isend(communication_send_1d_array, row_amount * M, MPI_FLOAT, dest, 2, MPI_COMM_WORLD, &reqs2);
			//receive index and rows
			MPI_Irecv(communication_receive_row_idx, row_amount, MPI_INT, src, 1, MPI_COMM_WORLD, &reqs1);
			MPI_Irecv(communication_receive_1d_array, row_amount * M, MPI_FLOAT, src, 2, MPI_COMM_WORLD, &reqs2);

			MPI_Wait (&reqs2, &status2);
			// set communication_receive_2d_array for following calculation
			int my_1d_matrix_count = 0;
			for(i = 0; i < row_amount; i++){
				for(j = 0; j < M; j++){
					communication_receive_2d_array[i][j] = communication_receive_1d_array[my_1d_matrix_count];
					my_1d_matrix_count++;
				}
			}
			int idx_count = 0;
			int test_count_i = 0;
			int test_count_j = 0;
			MPI_Wait (&reqs1, &status1);
			//after processor 0 received other rows, put the calculation results to final
			//i is for id 0's matrix, j is for received 2d matrix.
			for(i = 0; i < row_amount; i++){
				for(j = 0; j < row_amount; j++){
					int idx = result_index(i,communication_receive_row_idx[j],N);
					final[idx] = row_product(M,my_matrix[i], communication_receive_2d_array[j]);
					idx_count++;
				}
			}
			//get calculation results from other processors
			part_final_idx = (int*)calloc(row_amount*row_amount, sizeof(MPI_Request));
			part_final_value = (float*)calloc(row_amount*row_amount, sizeof(MPI_Request));
			//
			for(p_count = 1; p_count < numprocs; p_count++){
				//receive other calculation result, put them to final
				MPI_Irecv(part_final_idx, row_amount*row_amount, MPI_INT, p_count, 1, MPI_COMM_WORLD, &reqs1);
				MPI_Irecv(part_final_value, row_amount*row_amount, MPI_FLOAT, p_count, 2, MPI_COMM_WORLD, &reqs2);
				MPI_Wait (&reqs1, &status1);
				MPI_Wait (&reqs2, &status2);
				for(i = 0; i < row_amount*row_amount; i++){
					final[part_final_idx[i]] = part_final_value[i];
				}
			}
			sending_count = 0;
			i = 0;
			j = 0;
			for(sending_count = 0; sending_count < row_amount * M; sending_count++){
				communication_send_1d_array[sending_count] = communication_receive_1d_array[sending_count];
			}
			sending_count = 0;
			for(sending_count = 0; sending_count < row_amount; sending_count++){
				communication_send_row_idx[sending_count] = communication_receive_row_idx[sending_count];
			}
		}
		int final_count = 0 ;
		int current_element_amount = N-1;
		i = 1;
		printf("parallel result:\n" );
		for(final_count = 0; final_count < (N-1)*(1+N-1)/2; final_count++){
			printf("%f ",final[final_count] );
			if(final_count == current_element_amount-1){
				printf("\n" );
				current_element_amount += N-1-i;
				i++;
			}
		}
		printf("\nsequential result:\n" );
		print_non_block(N, M, matrix);
		//self checking part,similar to print_non_block() function
		final_count = 0 ;
		int self_checking_flag = 0;
		for(i = 0; i< N; i++){
			for(i1 = i+1; i1 < N; i1++ ){
				float sum = 0;
				for(j = 0; j<M; j++){
					sum += matrix[i][j]*matrix[i1][j];
				}
				if( final[final_count] != sum){
					self_checking_flag = 1;
					printf("something wrong! at (row:%d, column:%d), stop checking.\n", i,i1);
					break;
				}
				final_count++;
			}
			if (i == N-1){
				break;
			}
		}
		if(self_checking_flag == 0){
			printf("all good!\n");
		}

		free(final);
		free(communication_send_1d_array);
		free(communication_send_row_idx);
		free(communication_receive_1d_array);
		free(communication_receive_row_idx);
		free(communication_receive_2d_array);
		free(part_final_idx);
		free(part_final_value);
	}
	if(myid != 0){
		float my_1d_matrix[row_amount * M];
		MPI_Request request;
		MPI_Request reqs1;
		MPI_Request reqs2;
		MPI_Status status1;
		MPI_Status status2;
		//receive the data
		MPI_Irecv(my_1d_matrix, row_amount * M, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request);
		//set 2d array
		float matrix[row_amount][M];
		int my_1d_matrix_count = 0;
		MPI_Wait (&request, &status);
		for(i = 0; i < row_amount; i++){
			for(j = 0; j < M; j++){
				matrix[i][j] = my_1d_matrix[my_1d_matrix_count];
				my_1d_matrix_count++;
			}
		}
		int sending_count = 0;
		if(row_amount != 1){
			//the first calculation(calculate itself), size:(row_amount-1)*(1+row_amount-1)/2
			int result_amount = (row_amount-1)*(row_amount)/2;
			//to send first calculation to processor 0
			int sending_index[result_amount];
			float sending_result[result_amount];
			//real row number is myid*row_amount+index
			for(i = 0; i < row_amount-1; i++){
				for(i1 = i+1; i1 < row_amount; i1++ ){
					int real_i = (myid*row_amount+i);
					int real_i1 = (myid*row_amount+i1);
					sending_index[sending_count] = real_i*N -real_i*(1+real_i)/2 + real_i1-real_i-1;
					sending_result[sending_count] = row_product(M,matrix[i], matrix[i1]);
					sending_count++;
				}
			}
			//send sending_index, sending_result
			MPI_Isend(sending_index, result_amount, MPI_INT, 0, 1, MPI_COMM_WORLD, &reqs1);
			MPI_Isend(sending_result, result_amount, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &reqs2);
		}

		// communication and iteration stuff
		//for sending 1d array and index
		float *communication_send_1d_array=(float*)malloc(sizeof(float)*row_amount * M);
		int *communication_send_row_idx=(int*)malloc(sizeof(int)*row_amount);
		//for receiving 1d array and index
		float *communication_receive_1d_array=(float*)malloc(sizeof(float)*row_amount * M);
		int *communication_receive_row_idx=(int*)malloc(sizeof(int)*row_amount);
		//for changing 1d array to 2d array
		float **communication_receive_2d_array = (float **)malloc(row_amount * sizeof(float *));
		for (i = 0; i < row_amount; i++) {
			communication_receive_2d_array[i] = (float *)malloc(M * sizeof(float));
		}
		//set calculation results and indexes, send them to processor 0
		float *results=(float*)malloc(sizeof(float)*row_amount*row_amount);
		int *results_idx=(int*)malloc(sizeof(int)*row_amount*row_amount);
		//set initial status for sending array and index
		sending_count = 0;
		for(sending_count = 0; sending_count < row_amount * M; sending_count++){
			communication_send_1d_array[sending_count] = my_1d_matrix[sending_count];
		}
		sending_count = 0;
		for(sending_count = 0; sending_count < row_amount; sending_count++){
			communication_send_row_idx[sending_count] = myid*row_amount+sending_count;
		}
		int iter = 0;
		int dest;
		if(myid == numprocs-1){
			dest = 0;
		} else {
			dest = myid+1;
		}
		int src;
		src = myid - 1;
		for(iter = 0; iter < (numprocs-1)/2 ; iter++){
			//send data array and index
			MPI_Isend(communication_send_row_idx, row_amount, MPI_INT, dest, 1, MPI_COMM_WORLD, &reqs1);
			MPI_Isend(communication_send_1d_array, row_amount * M, MPI_FLOAT, dest, 2, MPI_COMM_WORLD, &reqs2);
			//receive data array and index
			MPI_Irecv(communication_receive_row_idx, row_amount, MPI_INT, src, 1, MPI_COMM_WORLD, &reqs1);
			MPI_Irecv(communication_receive_1d_array, row_amount * M, MPI_FLOAT, src, 2, MPI_COMM_WORLD, &reqs2);
			my_1d_matrix_count = 0;
			MPI_Wait (&reqs2, &status2);
			//change 1d array to 2d array
			for(i = 0; i < row_amount; i++){
				for(j = 0; j < M; j++){
					communication_receive_2d_array[i][j] = communication_receive_1d_array[my_1d_matrix_count];
					my_1d_matrix_count++;
				}
			}
			//do product between two processor(the matrix this processor has and the received matrix)
			int idx_count = 0;
			int test_count_i = 0;
			int test_count_j = 0;
			MPI_Wait (&reqs1, &status1);
			//set calculation result and index
			for(i = 0; i < row_amount; i++){
				for(j = 0; j < row_amount; j++){
					int idx = result_index((myid*row_amount+i),communication_receive_row_idx[j],N);
					results_idx[idx_count] = idx;
					results[idx_count] = row_product(M, matrix[i], communication_receive_2d_array[j]);
					idx_count++;
				}
			}
			//send calculation result and index to processor 0
			MPI_Isend(results_idx, row_amount*row_amount, MPI_INT, 0, 1, MPI_COMM_WORLD, &reqs1);
			MPI_Isend(results, row_amount*row_amount, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &reqs2);
			// set new communication_send_row_idx, communication_send_1d_array for next iteration
			sending_count = 0;
			for(sending_count = 0; sending_count < row_amount * M; sending_count++){
				communication_send_1d_array[sending_count] = communication_receive_1d_array[sending_count];
			}
			sending_count = 0;
			for(sending_count = 0; sending_count < row_amount; sending_count++){
				communication_send_row_idx[sending_count] = communication_receive_row_idx[sending_count];
			}
		}
		free(communication_send_1d_array);
		free(communication_send_row_idx);
		free(communication_receive_1d_array);
		free(communication_receive_row_idx);
		free(communication_receive_2d_array);
		free(results);
		free(results_idx);
	}
	MPI_Finalize();
    return 0;
}

int result_index(int index1, int index2, int N){
	int result;
	if(index1 < index2){
		result = (index1*N -index1*(1+index1)/2 + index2-index1-1);
	}else{
		result = (index2*N -index2*(1+index2)/2 + index1-index2-1);
	}
	return result;
}
//set matrix
void matrix_init(int r, int c, float matrix[r][c]){
	int i = 0, j = 0;
	for (i = 0; i < r; i++){
		for(j = 0; j < c; j++){
			matrix[i][j] = (float)rand() / (float)RAND_MAX;
		}
	}
}
float row_product(int c, float row1[c], float row2[c]){
	int i = 0;
	float sum = 0;
	for(i = 0; i < c; i++){
		sum += row1[i] * row2[i];
	}
	return sum;
}
void print_non_block(int r, int c, float matrix[r][c]){
	int i = 0;
	int i1 = 0;
	int j = 0;
	for(i = 0; i< r; i++){
		for(i1 = i+1; i1 < r; i1++ ){
			float sum = 0;
			for(j = 0; j<c; j++){
				sum += matrix[i][j]*matrix[i1][j];
			}
			printf("%f ",sum );
		}
		if (i == r-1){
			break;
		}
		printf("\n");
	}
}
