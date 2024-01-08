#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sched.h>


//======================
#define DEV_NO 0

int n, m;
int **Dist;
int INF = 1000000000;
int num_threads;


void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    Dist = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; ++i) {
        Dist[i] = (int *)malloc(n * sizeof(int));
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }


void freeDist(){
    for(int i = 0; i < n; ++i){
        free(Dist[i]);
    }
    free(Dist);
}

void Floyd_Warshall(){
    for(int k = 0; k < n; k ++){
#pragma omp parallel for num_threads(num_threads)  schedule(dynamic, 1)
        for(int i = 0; i < n; i ++){
            for(int j = 0; j < n; j++){
                if(Dist[i][k] + Dist[k][j] < Dist[i][j]){
                    Dist[i][j] = Dist[i][k] + Dist[k][j];
                }
            }
        }
    }
}



int main(int argc, char* argv[]) {
    input(argv[1]);
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = omp_get_max_threads();

    // block_FW(B);
    Floyd_Warshall();
    output(argv[2]);
    freeDist();
    return 0;
}