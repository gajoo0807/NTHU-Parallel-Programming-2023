#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
//#include <iostream>
//#include <smmintrin.h>
//#include <pmmintrin.h>
#include <emmintrin.h>  

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int *image;
int task_id;
pthread_mutex_t mutex;
int iters;
double left;
double right;
double lower;
double upper;
int width;
int height;



void* count_png(void *arg){
    double y0;
    __m128d ymm0, xmm0, repeatsmm, xmm, ymm, length_squaredmm, temp, mask, xymul;
    __m128d four = _mm_set1_pd(4.0);
    double x0_1, x0_2;
    int id;
    int maskResult;
    while(1){
        // 從 task_id 取得任務
        pthread_mutex_lock(&mutex);
        if(task_id >= height){
            pthread_mutex_unlock(&mutex);
            break;
        }else{
            id = task_id;
            task_id++;
        }
        pthread_mutex_unlock(&mutex);
        
        y0 = id * ((upper - lower) / height) + lower;
        ymm0 = _mm_set1_pd(y0);
        // int iter;
        double base = (right - left) / width;
        int base_address = id * width ;
        
        for (int iter = 0 ; iter < width - 1; iter += 2) {
            xmm0 = _mm_set_pd((iter + 1) * base + left, iter * base + left);

            xmm = _mm_setzero_pd();
            ymm = _mm_setzero_pd();
            length_squaredmm = _mm_setzero_pd();
            int repeats[2] = {1, 1};
            // 進行迭代，每一個iteration先計算這一輪的xmm、ymm，再去更新repeatsmm
            while(repeats[0] < iters && repeats[1] < iters){
            // for(int k = 1; k < iters; k++){
                xymul = _mm_mul_pd(xmm, ymm);
                temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(xmm, xmm), _mm_mul_pd(ymm, ymm)), xmm0); 
                ymm = _mm_add_pd(_mm_add_pd(xymul, xymul), ymm0); 
                xmm = temp;

                length_squaredmm = _mm_add_pd(_mm_mul_pd(xmm, xmm), _mm_mul_pd(ymm, ymm)); 


                mask = _mm_cmplt_pd(length_squaredmm, four);

                maskResult = _mm_movemask_pd(mask);
                if(!maskResult)
                    break;
                repeats[0] += (maskResult & 1);
                repeats[1] += (maskResult >> 1) & 1;


            }
            memcpy(&image[id * width + iter], repeats, 2 * sizeof(int));
        }


        if(width  & 1){
            double x0 = ((width - 1) * ((right - left) / width)) + left;
            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) { 
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            // image[id * width + iter] = repeats;
            image[id * width + width - 1] = repeats;
        }
    }
    pthread_exit(NULL);
}


int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int n_cpu = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1]; 
    iters = strtol(argv[2], 0, 10); // number of iterations
    left = strtod(argv[3], 0); // inclusive/ non-inclusive x range
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0); // inclusive/ non-inclusive y range
    upper = strtod(argv[6], 0); 
    width = strtol(argv[7], 0, 10); // number of points in the x-axis for output
    height = strtol(argv[8], 0, 10); // number of points in the y-axis for output

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    pthread_t thread[n_cpu];
    task_id = 0;
    pthread_mutex_init(&mutex, NULL);


    for(int k = 0; k < n_cpu; k++){
        pthread_create(&thread[k], NULL, count_png, NULL);
    }
    for(int k = 0; k < n_cpu; k++){
        pthread_join(thread[k], NULL);
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
    pthread_mutex_destroy(&mutex);
}