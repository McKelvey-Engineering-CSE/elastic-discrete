#include "task.h"
#include <immintrin.h>

#define N 100

int mat1[N][N];
int mat2[N][N];
int result[N][N];

int init(int argc, char *argv[])
{

	std::cout << "Initializing Arrays" << std::endl;

	//give some random values
	for (int x = 0; x < N; x++){
		for (int y = 0; y < N; y++){
			mat1[x][y] = x+y;
			mat2[x][y] = int(x/2)+int(y/4);
			result[x][y] = 0;
		}
	}

	return 0;       
}

int run(int argc, char *argv[]){
	//*(int * ) 0 = 0;

	std::cout << "Executing Matrix Manipulations" << std::endl;

	//Example Vector workload
	__m256i vec_multi_res = _mm256_setzero_si256();
	__m256i vec_mat1 = _mm256_setzero_si256();
	__m256i vec_mat2 = _mm256_setzero_si256();

	int i, j, k;
	for (i = 0; i < N; i++){
		for (j = 0; j < N; ++j){
			//Stores one element in mat1 and use it in all computations needed before proceeding
			//Stores as vector to increase computations per cycle
			vec_mat1 = _mm256_set1_epi32(mat1[i][j]);

			for (k = 0; k < N; k += 8){
				vec_mat2 = _mm256_loadu_si256((__m256i*)&mat2[j][k]);
				vec_multi_res = _mm256_loadu_si256((__m256i*)&result[i][k]);
				vec_multi_res = _mm256_add_epi32(vec_multi_res ,_mm256_mullo_epi32(vec_mat1, vec_mat2));
			}
		}
	}

	return 0;
}

int finalize(int argc, char *argv[])
{
    return 0;
}

task_t task = { init, run, finalize };
