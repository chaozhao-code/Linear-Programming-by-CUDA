#include "floating_number_helper.h"
#include "input_output.h"
#include <float.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include "math.h"
#include <iostream>
#include "device_launch_parameters.h"

#include "PerformanceTimer.h"

PerformanceTimer timer;
#define THREAD_NUM 256

//
// this function is what you need to finish
// @Usage : to solve the problem
// @Input : input containing all data needed
// @Output: answer containing all necessary data
//  you can find the definition of the two structs above in
//      input_output.h
//

answer * compute(inputs * input) {
	int num = input->number;
	line ** lines = input->lines;
	answer * ans = (answer *)malloc(sizeof(answer));
	double tmp = 0;
	//double min = MAXFLOAT;
	double min = FLT_MAX;
	for (int i = 0; i < num - 1; i++) {
		line * now_line = lines[i];
		for (int j = i + 1; j < num; j++) {
			line * tmp_line = lines[j];
			point * tmp_node = generate_intersection_point(now_line, tmp_line);
			//std::cout << tmp_node->pos_x << "   " << tmp_node->pos_y << std::endl;
			tmp = input->obj_function_param_a * tmp_node->pos_x + input->obj_function_param_b * tmp_node->pos_y;
			//std::cout << tmp << std::endl;
			if (tmp <= min) {
				for (int k = 0; k < num; k++) {
					if ((tmp_node->pos_x * lines[k]->param_a + tmp_node->pos_y * lines[k]->param_b) < lines[k]->param_c) {
						break;
					}
					else if (k == num - 1) {
						//std::cout << "we can get answer" << std::endl;
						min = tmp;
						ans->answer_b = tmp;
						ans->intersection_point = tmp_node;
						ans->line1 = now_line;
						ans->line2 = tmp_line;
					}
				}
			}
		}
	}
	return ans;
}

__device__ int equals_gpu(double num1, double num2) {
	return fabs(num1 - num2) < EPS ? TRUE : FALSE;
}

__device__ int is_parallel_gpu(line * line1, line * line2) {
	return equals_gpu(line1->param_a * line2->param_b, line1->param_b * line2->param_a);
}

__device__ point * generate_intersection_point_gpu(line * line1, line * line2) {
	if (is_parallel_gpu(line1, line2)) {
		return NULL;
	}
	point * new_point = (point *)malloc(sizeof(point));
	new_point->pos_x = (line1->param_c * line2->param_b - line1->param_b * line2->param_c)
		/ (line1->param_a * line2->param_b - line1->param_b * line2->param_a);
	new_point->pos_y = (line1->param_c * line2->param_a - line1->param_a * line2->param_c)
		/ (line1->param_b * line2->param_a - line1->param_a * line2->param_b);
	return new_point;
}

__global__ static void gpu_compute(int * dev_num, double * dev_param_a, 
	         double * dev_param_b, line * dev_line_array, for_answer * dev_potential) //, double* arr) 
{
	int num = *dev_num;
	double a = *dev_param_a;
	double b = *dev_param_b;
	double min = FLT_MAX;
	int line2 = 0 ;
	//int i = threadIdx.x; 
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < num)
	{
		for (int j = i + 1; j < num; j++)
		{
			point * tmp_node = generate_intersection_point_gpu(&dev_line_array[i], &dev_line_array[j]);
			double tmp = a * tmp_node->pos_x + b * tmp_node->pos_y;
			//printf("%d  %d  %f\n", i, j, tmp);
			for (int k = 0; k < num; k++)
			{
				if ((tmp_node->pos_x * dev_line_array[k].param_a + tmp_node->pos_y * dev_line_array[k].param_b) < dev_line_array[k].param_c - 0.0001)
				{
					//printf("%f", tmp);
					break;
				}
				else if (k == num - 1 && tmp <= min)
				{
					min = tmp;
					line2 = j;
				}
			}
		}
		 //printf("%d  %d  %f\n", i, line2, min);
		dev_potential[i].line1 = threadIdx.x;
		dev_potential[i].line2 = line2;
		dev_potential[i].possible = min;
	}

}

void checkCUDAError() {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(error));
	}
}

int main() {
	cudaError_t err = cudaSuccess;

	// 1. get the input data
	inputs * input = read_from_file("C:/Users/hzauz/Desktop/sws3003_assignment 1/test_cases/1_0.dat");
	int num = input->number;
	
	for_answer * potential = (for_answer*)malloc(sizeof(for_answer) * num);
	line * line_array = (line*)malloc(sizeof(line) * num);
	if (line_array == NULL || potential == NULL)
	{
		fprintf(stderr, "Failed to allocate host input data!\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < num; i++)
	{
		potential[i].possible = FLT_MAX;
		// debug
		potential[i].line1 = i;
	}
                            
	for (int i = 0; i < num; i++)
	{
		line_array[i] = *((input->lines)[i]);
	}

	for_answer * dev_potential = NULL;
	err = cudaMalloc((void**)&dev_potential, sizeof(for_answer) * num);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device potential vector (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(dev_potential, potential, sizeof(for_answer) * num, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy potential vector from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	line * dev_line_array = NULL;
	err = cudaMalloc((void**)&dev_line_array, sizeof(line) * num);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device line (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(dev_line_array, line_array, sizeof(line) * num, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy line array from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	int * dev_num;                                                                      // for input the number of line
	cudaMalloc((void**)&dev_num, sizeof(int));
	cudaMemcpy(dev_num, &num, sizeof(int), cudaMemcpyHostToDevice);

	double * dev_param_a;                                                               // for input the number of a and b
	double param_a = input->obj_function_param_a;
	cudaMalloc((void**)&dev_param_a, sizeof(double));
	cudaMemcpy(dev_param_a, &param_a, sizeof(double), cudaMemcpyHostToDevice);

	double * dev_param_b;
	double param_b = input->obj_function_param_b;
	cudaMalloc((void**)&dev_param_b, sizeof(double));
	cudaMemcpy(dev_param_b, &param_b, sizeof(double), cudaMemcpyHostToDevice);
	
	// 3.computing by using gpu
	int blocks_num = (num + THREAD_NUM - 1) / THREAD_NUM;
	//gpu_compute<<<blocks_num,THREAD_NUM>>>(dev_num, dev_param_a, dev_param_b, dev_line_array, dev_potential);
	int threadsPerBlock = 256;
	int blocksPerGrid = (num + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	timer.StartTimer();
	gpu_compute << <blocksPerGrid, threadsPerBlock >> >(dev_num, dev_param_a, dev_param_b, dev_line_array, dev_potential);
	printf("Processing time (GPU): %f (ms) \n", timer.GetTimeElapsed() * 1000);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch Linear programmer kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}



	// 4.getting the final answer
	err = cudaMemcpy(potential, dev_potential, sizeof(for_answer) * num, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy potential from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double min = FLT_MAX;
	int minidx = 0;
	for (int i = 0; i < num; i++)
	{
		//std::cout << potential[i].line1 << "   " << potential[i].line2 << std::endl;
		if (potential[i].possible < min - 0.0001)
		{
			min = potential[i].possible;
			minidx = i;
		}
	}
	
	//std::cout << "the minidx is :" << minidx << std::endl;
	//std::cout << potential[minidx].line1 << " " << potential[minidx].line2 << std::endl;

	if (minidx == 0 && min - FLT_MAX < 0.1 && min - FLT_MAX > -0.1)
	{
		printf("Failed to calculate the answer.\n");
		exit(EXIT_FAILURE);
	}

	answer * ans = (answer *)malloc(sizeof(answer));
	point * tmp_node = generate_intersection_point(&line_array[potential[minidx].line1], &line_array[potential[minidx].line2]);
	if (tmp_node == NULL)
	{
		std::cout << "No resolution" << std::endl;
		return 0;
	}
	
	
	double tmp = param_a * tmp_node->pos_x + param_b * tmp_node->pos_y;
	ans->answer_b = tmp;
	ans->intersection_point = tmp_node;
	ans->line1 = input->lines[potential[minidx].line1];
	ans->line2 = input->lines[potential[minidx].line2];
	// 3. display result and free memory
	char * ans_string = generate_ans_string(ans);
	printf("%s", ans_string);
	free_inputs(&input);
	cudaFree(dev_line_array);
	free(line_array);
	free_ans(&ans);
	free(ans_string);
	cudaFree(dev_potential);
	free(potential);

	return 0;
}

