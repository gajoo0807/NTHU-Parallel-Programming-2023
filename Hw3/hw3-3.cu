#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <omp.h>



//======================
#define DEV_NO 0
#define block_size 64
#define thread_num 32


int n, m, N;
int *Dist;
const int INF = ((1 << 30) - 1);
void input(char* inFileName);
void output(char* outFileName);

// void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);


void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    size_t bytesRead = fread(&n, sizeof(int), 1, file);
    bytesRead = fread(&m, sizeof(int), 1, file);

	if (n % block_size) N = n + (block_size - n % block_size);
	else N = n;

    Dist = (int *)malloc(N * N * sizeof(int));
    #pragma unroll 64
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Dist[i * N + j] = INF;
            if (i == j)
                Dist[i * N + j] = 0;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        // Dist[pair[0]][pair[1]] = pair[2];
        Dist[pair[0] * N + pair[1]] = pair[2];
    }


    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    #pragma unroll 32
    for(int i = 0; i < n; i++){
        fwrite(&Dist[i * N], sizeof(int), n, outfile);
    }
    fclose(outfile);
}




__global__ void phase1(int r, int *deviceDist, int N){
    int i = threadIdx.y , j = threadIdx.x;
    int offset = r * block_size;
    __shared__ int share[block_size][block_size];
    int addr_self = (i + offset) * N + (j + offset), addr_right = (i + offset) * N + (j + 32 + offset), addr_down = (i + 32+ offset) * N + (j + offset), addr_down_right = (i + 32 + offset) * N + (j + 32+ offset);
    share[i][j] = deviceDist[addr_self], share[i][j+32] = deviceDist[addr_right], share[i+32][j] = deviceDist[addr_down], share[i+32][j+32] = deviceDist[addr_down_right];

    // printf("i_start: %d, j_start: %d\n", i_start, j_start);
    #pragma unroll 64
    for(int k = 0; k < block_size; k ++){
        __syncthreads();
        share[i][j] = min(share[i][j], share[i][k] + share[k][j]);
        share[i + 32][j] = min(share[i + 32][j], share[i + 32][k] + share[k][j]);
        share[i][j + 32] = min(share[i][j + 32], share[i][k] + share[k][j + 32]);
        share[i + 32][j + 32] = min(share[i + 32][j + 32], share[i + 32][k] + share[k][j + 32]);
    }
    deviceDist[addr_self] = share[i][j], deviceDist[addr_right] = share[i][j+32], deviceDist[addr_down] = share[i+32][j], deviceDist[addr_down_right] = share[i+32][j+32];
}


__global__ void phase2(int r, int *deviceDist, int N){
    int i = threadIdx.y, j = threadIdx.x;
    
    // int i_offset, j_offset;
    
    __shared__ int pivot_block[block_size][block_size]; 
    __shared__ int Ver_block[block_size][block_size];
    __shared__ int Hor_block[block_size][block_size];
    // x ==0 ，直的
    int offset = block_size * r;
    int Ver_i_offset, Hor_j_offset;
    Ver_i_offset = (blockIdx.x < r) ? blockIdx.x * block_size : (blockIdx.x + 1) * block_size;
    Hor_j_offset = (blockIdx.x < r) ? blockIdx.x * block_size : (blockIdx.x + 1) * block_size;
    // int pivot_addr, Ver_addr, Hor_addr;


    int pivot_addr_self = (i + offset) * N + (j + offset), pivot_addr_right = (i + offset) * N + (j + 32 + offset), pivot_addr_down = (i +32 + offset) * N + (j + offset), pivot_addr_down_right = (i +32 + offset) * N + (j + 32 + offset);
    int Ver_addr_self = (i + Ver_i_offset)* N + (j + offset), Ver_addr_right = (i + Ver_i_offset)* N + (j + 32 + offset), Ver_addr_down = (i + 32 + Ver_i_offset)* N + (j + offset), Ver_addr_down_right = (i + 32 + Ver_i_offset)* N + (j + 32 + offset);
    int Hor_addr_self = (i + offset)* N+ (j + Hor_j_offset), Hor_addr_right = (i + offset)* N + (j + 32 + Hor_j_offset), Hor_addr_down = (i + 32+ offset)* N + (j + Hor_j_offset), Hor_addr_down_right = (i + 32+ offset)* N + (j + 32+ Hor_j_offset);

    pivot_block[i][j] = deviceDist[pivot_addr_self], pivot_block[i][j+ 32] = deviceDist[pivot_addr_right], pivot_block[i+32][j] = deviceDist[pivot_addr_down],pivot_block[i+32][j+32] = deviceDist[pivot_addr_down_right];
    Ver_block[i][j] = deviceDist[Ver_addr_self], Ver_block[i][j+ 32] = deviceDist[Ver_addr_right], Ver_block[i+32][j] = deviceDist[Ver_addr_down], Ver_block[i+32][j+32] = deviceDist[Ver_addr_down_right];
    Hor_block[i][j] = deviceDist[Hor_addr_self], Hor_block[i][j+32] = deviceDist[Hor_addr_right], Hor_block[i+32][j] = deviceDist[Hor_addr_down], Hor_block[i+32][j+32] = deviceDist[Hor_addr_down_right];
    __syncthreads();
    // #pragma unroll 64
    for(int k = 0; k < block_size; k++){
        
        Ver_block[i][j] = min(Ver_block[i][j], Ver_block[i][k] + pivot_block[k][j]);
        Ver_block[i + 32][j] = min(Ver_block[i + 32][j], Ver_block[i + 32][k] + pivot_block[k][j]);
        Ver_block[i][j + 32] = min(Ver_block[i][j + 32], Ver_block[i][k] + pivot_block[k][j + 32]);
        Ver_block[i + 32][j + 32] = min(Ver_block[i + 32][j + 32], Ver_block[i + 32][k] + pivot_block[k][j + 32]);

        Hor_block[i][j] = min(Hor_block[i][j], pivot_block[i][k] + Hor_block[k][j]);
        Hor_block[i + 32][j] = min(Hor_block[i + 32][j], pivot_block[i + 32][k] + Hor_block[k][j]);
        Hor_block[i][j + 32] = min(Hor_block[i][j + 32], pivot_block[i][k] + Hor_block[k][j + 32]);
        Hor_block[i + 32][j + 32] = min(Hor_block[i + 32][j + 32], pivot_block[i + 32][k] + Hor_block[k][j + 32]);
    }
    deviceDist[Ver_addr_self] = Ver_block[i][j], deviceDist[Ver_addr_right] = Ver_block[i][j+32], deviceDist[Ver_addr_down] = Ver_block[i+32][j], deviceDist[Ver_addr_down_right] = Ver_block[i+32][j+32];
    deviceDist[Hor_addr_self] = Hor_block[i][j], deviceDist[Hor_addr_right] = Hor_block[i][j+32], deviceDist[Hor_addr_down] = Hor_block[i+32][j], deviceDist[Hor_addr_down_right] = Hor_block[i+32][j+32];
}
__global__ void phase3(int r, int *deviceDist, int N, int row_offset){
    int i = threadIdx.y , j = threadIdx.x;
    int i_offset, j_offset, i_block, j_block;
    int offset = r * block_size;

    // printf("offset: %d\n", offset);
    i_block = blockIdx.y + row_offset;
    j_block = blockIdx.x;
    if(i_block == r || j_block == r) return;

    i_offset = i_block * block_size;
    j_offset = j_block * block_size;

    __shared__ int my_block[block_size][block_size];
    __shared__ int Hor_block[block_size][block_size]; // 橫的block
    __shared__ int Ver_block[block_size][block_size]; // 縱的block


    // int my_block_addr, Hor_block_addr, Ver_block_addr;
    int my_block_addr_self = (i + i_offset) * N + (j + j_offset), my_block_addr_right = (i + i_offset) * N + (j + 32 + j_offset), my_block_addr_down = (i + 32 + i_offset) * N + (j + j_offset), my_block_addr_down_right = (i + 32 + i_offset) * N + (j + 32+ j_offset);
    int Hor_block_addr_self = (i + offset) * N + (j + j_offset), Hor_block_addr_right = (i + offset) * N + (j + 32 + j_offset), Hor_block_addr_down = (i + 32 + offset) * N + (j + j_offset), Hor_block_addr_down_right = (i + 32 + offset) * N + (j + 32 + j_offset);
    int Ver_block_addr_self = (i + i_offset) * N + (j + offset), Ver_block_addr_right = (i + i_offset) * N + (j + 32 + offset), Ver_block_addr_down = (i + 32 + i_offset) * N + (j + offset), Ver_block_addr_down_right = (i + 32 + i_offset) * N + (j + 32 + offset);


    my_block[i][j] = deviceDist[my_block_addr_self], my_block[i][j+32] = deviceDist[my_block_addr_right], my_block[i+32][j] = deviceDist[my_block_addr_down], my_block[i+32][j+32] = deviceDist[my_block_addr_down_right];
    Hor_block[i][j] = deviceDist[Hor_block_addr_self], Hor_block[i][j+32] = deviceDist[Hor_block_addr_right], Hor_block[i+32][j] = deviceDist[Hor_block_addr_down], Hor_block[i+32][j+32] = deviceDist[Hor_block_addr_down_right];
    Ver_block[i][j] = deviceDist[Ver_block_addr_self], Ver_block[i][j+32] = deviceDist[Ver_block_addr_right], Ver_block[i+32][j] = deviceDist[Ver_block_addr_down], Ver_block[i+32][j+32] = deviceDist[Ver_block_addr_down_right];

    __syncthreads();
    // #pragma unroll 64
    for(int k = 0; k < block_size; k++){

        my_block[i][j] = min(my_block[i][j], Ver_block[i][k] + Hor_block[k][j]);
        my_block[i + 32][j] = min(my_block[i + 32][j], Ver_block[i + 32][k] + Hor_block[k][j]);
        my_block[i][j + 32] = min(my_block[i][j + 32], Ver_block[i][k] + Hor_block[k][j + 32]);
        my_block[i + 32][j + 32] = min(my_block[i + 32][j + 32], Ver_block[i + 32][k] + Hor_block[k][j + 32]);
    }
    deviceDist[my_block_addr_self] = my_block[i][j], deviceDist[my_block_addr_right] = my_block[i][j+32], deviceDist[my_block_addr_down] = my_block[i+32][j], deviceDist[my_block_addr_down_right] = my_block[i+32][j+32];
}

int main(int argc, char* argv[]) {

    input(argv[1]); // Read Dist並將其copy到deviceDist
    int round = (N + block_size - 1) / block_size;
    int *deviceDist[2];
    cudaEvent_t start_t, stop_t;
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);
    cudaEventRecord(start_t, 0);

    
    #pragma omp parallel num_threads(2)
    {

        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int nei_cpu_thread_id = !cpu_thread_id;
        cudaSetDevice(cpu_thread_id);
        cudaDeviceEnablePeerAccess(nei_cpu_thread_id, 0);

        unsigned int start_block =(cpu_thread_id == 1)? round / 2: 0;
        unsigned int job_size = (cpu_thread_id == 0) ? round / 2 : round - round / 2;
        unsigned int start_offset = start_block * block_size * N;
        unsigned int length_byte = job_size * block_size * N * sizeof(int);
        unsigned int pivot_offset;
        unsigned int length_per_row_byte = block_size * N * sizeof(int);


        dim3 block(thread_num, thread_num);
        dim3 block3(round, job_size);

        // cudaMalloc((void **)&deviceDist[cpu_thread_id], N * N * sizeof(int));
        cudaMalloc(&deviceDist[cpu_thread_id], N*N*sizeof(int));
        cudaMemcpy(deviceDist[cpu_thread_id] + start_offset, Dist + start_offset, length_byte, cudaMemcpyHostToDevice);
        // cudaMemcpy(deviceDist[cpu_thread_id], Dist, N * N * sizeof(int), cudaMemcpyHostToDevice);

#pragma omp barrier
        for(int r = 0; r < round ; r ++){
            if(r >= start_block && r < (start_block + job_size)){
                pivot_offset = r * block_size * N;
                cudaMemcpy(deviceDist[nei_cpu_thread_id] + pivot_offset, deviceDist[cpu_thread_id] + pivot_offset, length_per_row_byte, cudaMemcpyDefault);
            }
#pragma omp barrier
            phase1<<<1, block>>>(r, deviceDist[cpu_thread_id], N);
            phase2<<<round - 1, block>>>(r, deviceDist[cpu_thread_id], N);
            phase3<<<block3, block>>>(r, deviceDist[cpu_thread_id], N, start_block);
        }
        cudaMemcpy(Dist + start_offset, deviceDist[cpu_thread_id] + start_offset, length_byte, cudaMemcpyDeviceToHost);
        cudaFree(deviceDist[cpu_thread_id]);
    }
    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
    printf("Total Time: %f ms\n", elapsedTime);
    output(argv[2]);
    free(Dist);
    return 0;
}