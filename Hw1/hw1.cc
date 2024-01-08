#include <cstdio>
#include <cstdlib>
#include <mpi.h>
// #include <boost/sort/spreadsort/detail/float_sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <iostream>
#include <queue>


void Small_Sort(float *my_data,float *received_data, float *result, int data_size, int received_data_size){
    int i = 0, j = 0, k = 0;
    while(k < data_size && i < data_size && j < received_data_size){
        if(my_data[i] <= received_data[j]){
            result[k++] = my_data[i++];
        }else{
            result[k++] = received_data[j++];
        }
    }
    if(i == data_size){
        while(k < data_size){
            result[k++] =  received_data[j++];
        }
    }else{
        while(k < data_size){
            result[k++] =  my_data[i++];
        }
    }
}
void Big_Sort(float *my_data, float *received_data, float *result, int data_size, int received_data_size){
    int i = data_size -1, j = received_data_size - 1, k = data_size - 1;
    
    while(k >= 0 && i >= 0 && j >= 0){
        if(my_data[i] >= received_data[j]){
            result[k--] = my_data[i--];
        }else{
            result[k--] = received_data[j--];
        }
    }
    if(i == 0){
        while(k >= 0){
            result[k--] =  received_data[j--];
        }        
    }else{
        while(k >= 0){
            result[k--] =  my_data[i--];
        }
    }
}
void Swap(int rank, float *my_data, int n, int odd, float *received_data, float *swaping_space, int remaining_elements, int *mode){ // 0: even swap, 0和1 swap; 1: odd swap, 1和2 swap
    int dest_rank;
    int received_data_size;
    if((rank & 1) == (odd &1)){
        dest_rank = rank + 1;
        received_data_size = (rank == remaining_elements -1)? n -1 : n;


        if((*mode) ==0){
            MPI_Sendrecv(&my_data[n-1], 1, MPI_FLOAT, dest_rank, 0, received_data, 1, MPI_FLOAT, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if(my_data[n-1] <= received_data[0]){
                return;
            }
            MPI_Sendrecv(my_data, n, MPI_FLOAT, dest_rank, 0, received_data, received_data_size, MPI_FLOAT, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Small_Sort(my_data, received_data, swaping_space, n, received_data_size);
            (*mode) = 1;
        }else{
            MPI_Sendrecv(&swaping_space[n-1], 1, MPI_FLOAT, dest_rank, 0, received_data, 1, MPI_FLOAT, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(swaping_space[n-1] <= received_data[0]){
                return;
            }
            MPI_Sendrecv(swaping_space, n, MPI_FLOAT, dest_rank, 0, received_data, received_data_size, MPI_FLOAT, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Small_Sort(swaping_space, received_data, my_data, n, received_data_size);
            (*mode) = 0;
        }
    }else{
        dest_rank = rank - 1;
        if(rank == remaining_elements && remaining_elements != 0) received_data_size = n + 1;
        else received_data_size = n;

        if(((*mode)) ==0){
            MPI_Sendrecv(&my_data[0], 1, MPI_FLOAT, dest_rank, 0, received_data, 1, MPI_FLOAT, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(my_data[0] >= received_data[0]){
                return;
            }
            MPI_Sendrecv(my_data, n, MPI_FLOAT, dest_rank, 0, received_data, received_data_size, MPI_FLOAT, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Big_Sort(my_data, received_data, swaping_space, n, received_data_size);
            (*mode) = 1;
        }else{
            MPI_Sendrecv(&swaping_space[0], 1, MPI_FLOAT, dest_rank, 0, received_data, 1, MPI_FLOAT, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(swaping_space[0] >= received_data[0]){
                return;
            }

            MPI_Sendrecv(swaping_space, n, MPI_FLOAT, dest_rank, 0, received_data, received_data_size, MPI_FLOAT, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Big_Sort(swaping_space, received_data, my_data, n, received_data_size);
            (*mode) = 0;
        }
    }
}


int main(int argc, char **argv)
{


    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    char *a = argv[1];
    char *endptr;
    long arraySize=strtol(a, &endptr, 10);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    MPI_File input_file, output_file;
    

    int data_size = arraySize / size;
    int remaining_elements = arraySize % size; 
    int offset;

    
    float *received_data = (float *)malloc((data_size+1) * sizeof(float));

    if (rank < remaining_elements) {
        data_size += 1;
        offset = rank * data_size;
    }else{
        offset = remaining_elements*(data_size+1) + (rank - remaining_elements) * data_size;
    }
    float *data = (float *)malloc(data_size * sizeof(float));
    float *swaping_space = (float *)malloc((data_size) * sizeof(float));


    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * offset, data, data_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);


    boost::sort::spreadsort::detail::float_sort(data, data + data_size);
    int mode = 0;
    int isSizeOdd = size & 1;
    int swaping_round = (isSizeOdd || remaining_elements)? (size >> 1) : (size >> 1) - 1;

    for (int i = 0; i < swaping_round; i++) {
        if (!(isSizeOdd && rank == size - 1)) 
            Swap(rank, data, data_size, 0, received_data, swaping_space, remaining_elements, &mode); // even phase
        if (rank != 0 && !(rank == size - 1 && !isSizeOdd)) 
            Swap(rank, data, data_size, 1, received_data, swaping_space, remaining_elements, &mode); // odd phase
    }
    if (!(isSizeOdd && rank == size - 1)) 
        Swap(rank, data, data_size, 0, received_data, swaping_space, remaining_elements, &mode); // even phase


    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);

    if(mode == 0) {
        MPI_File_write_at(output_file, sizeof(float) * offset, data, data_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }else {
        MPI_File_write_at(output_file, sizeof(float) * offset, swaping_space, data_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&output_file);



    free(data);
    free(swaping_space);
    free(received_data);


    MPI_Finalize();

    return 0;
}