#include "task.h"

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
#endif

#define N 100

#if defined(__x86_64__) || defined(_M_X64)
	int mat1[N][N];
	int mat2[N][N];
	int result[N][N];

#elif defined(__aarch64__) || defined(_M_ARM64)
	uint8_t mat1[N][N];
	uint8_t mat2[N][N];
	uint8_t result[N][N];

#endif

int init(int argc, char *argv[])
{

	print_module::task_print(std::cout, "Initializing Arrays\n");

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

	/*if (getpid() % 2 == 0)
		modify_self(2);
	else
		modify_self(1);*/

	print_module::task_print(std::cout, "Executing Matrix Manipulations\n");

	#if defined(__x86_64__) || defined(_M_X64)
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

	#elif defined(__aarch64__) || defined(_M_ARM64)

		//can only do 128 bit vectors..... I think....?
		//make 3 128 bit vectors
		uint8x16x3_t vec_multi_res;
		
		int i, j, k;
		for (i = 0; i < N; i++){
			for (j = 0; j < N; ++j){
				//Stores one element in mat1 and use it in all computations needed before proceeding
				//Stores as vector to increase computations per cycle
				vec_multi_res.val[1] = vld1q_u8(mat1[i]);

				for (k = 0; k < N; k += 8){
					vec_multi_res.val[2] = vld1q_u8(mat2[j]);
					vec_multi_res.val[0] = vld1q_u8(result[i]);

					//mult -> add -> store
					vec_multi_res.val[0] = (vec_multi_res.val[1] + vec_multi_res.val[2]) * vec_multi_res.val[0];
					vst1q_u8(result[i], vec_multi_res.val[0]);
				}
			}
		}

	#endif



	return 0;
}

int finalize(int argc, char *argv[])
{
    return 0;
}

task_t task = { init, run, finalize };
