#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>



//======================
#define block_size 64
#define thread_num 32


int n, m, N;
int *d_n;
int *Dist, *deviceDist;
const int INF = ((1 << 30) - 1);
void input(char* inFileName);
void output(char* outFileName);

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
    size_t detect;
    for (int i = 0; i < m; ++i) {
        detect = fread(pair, sizeof(int), 3, file);
        // Dist[pair[0]][pair[1]] = pair[2];
        Dist[pair[0] * N + pair[1]] = pair[2];
    }

    cudaMalloc((void **)&d_n, sizeof(int));
    cudaMemcpy(d_n, &N, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&deviceDist, N * N * sizeof(int));
    cudaMemcpy(deviceDist, Dist, N * N * sizeof(int), cudaMemcpyHostToDevice);
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
    // 此部驗證沒問題
    
    int i = threadIdx.y , j = threadIdx.x;
    int offset = r * block_size;
    __shared__ int share[block_size][block_size];
    int addr_self = (i + offset) * N + (j + offset), addr_right = (i + offset) * N + (j + 32 + offset), addr_down = (i + 32+ offset) * N + (j + offset), addr_down_right = (i + 32 + offset) * N + (j + 32+ offset);
    share[i][j] = deviceDist[addr_self], share[i][j+32] = deviceDist[addr_right], share[i+32][j] = deviceDist[addr_down], share[i+32][j+32] = deviceDist[addr_down_right];

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
    int Hor_addr_self = (i + offset)* N + (j + Hor_j_offset), Hor_addr_right = (i + offset)* N + (j + 32 + Hor_j_offset), Hor_addr_down = (i + 32+ offset)* N + (j + Hor_j_offset), Hor_addr_down_right = (i + 32+ offset)* N + (j + 32+ Hor_j_offset);

    pivot_block[i][j] = deviceDist[pivot_addr_self], pivot_block[i][j+ 32] = deviceDist[pivot_addr_right], pivot_block[i+32][j] = deviceDist[pivot_addr_down],pivot_block[i+32][j+32] = deviceDist[pivot_addr_down_right];
    Ver_block[i][j] = deviceDist[Ver_addr_self], Ver_block[i][j+ 32] = deviceDist[Ver_addr_right], Ver_block[i+32][j] = deviceDist[Ver_addr_down], Ver_block[i+32][j+32] = deviceDist[Ver_addr_down_right];
    Hor_block[i][j] = deviceDist[Hor_addr_self], Hor_block[i][j+32] = deviceDist[Hor_addr_right], Hor_block[i+32][j] = deviceDist[Hor_addr_down], Hor_block[i+32][j+32] = deviceDist[Hor_addr_down_right];
    __syncthreads();
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
__global__ void phase3(int r, int *deviceDist, int N){
    int i = threadIdx.y , j = threadIdx.x;
    int i_offset, j_offset, i_block, j_block;
    int offset = r * block_size;

    // printf("offset: %d\n", offset);
    i_block = (blockIdx.x < r) ? blockIdx.x : (blockIdx.x + 1) ;
    j_block = (blockIdx.y < r) ? blockIdx.y  : (blockIdx.y + 1) ;

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
    dim3 block(thread_num, thread_num);
    dim3 block3(round - 1, round - 1);
    
    for(int r = 0; r < round; r++){
        phase1<<<1, block>>>(r, deviceDist, N);
        phase2<<<round - 1, block>>>(r, deviceDist, N);
        phase3<<<block3, block>>>(r, deviceDist, N);
    }

    cudaMemcpy(Dist, deviceDist, N * N * sizeof(int), cudaMemcpyDeviceToHost); // 將deviceDist copy回Dist
    output(argv[2]);
    free(Dist);
    cudaFree(d_n);
    return 0;
}