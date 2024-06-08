#include "floating_number_helper.h"
#include "input_output.h"
#include <float.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include "math.h"
#include <iostream>
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\transform.h>
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
			tmp = input->obj_function_param_a * tmp_node->pos_x + input->obj_function_param_b * tmp_node->pos_y;
			if (tmp <= min) {
				for (int k = 0; k < num; k++) {
					if ((tmp_node->pos_x * lines[k]->param_a + tmp_node->pos_y * lines[k]->param_b) < lines[k]->param_c) {
						break;
					}
					else if (k == num - 1) {
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

__global__ static void gpu_compute(int * dev_num, double * dev_param_a, double * dev_param_b, line * dev_line_array, double * valmax) {
	int num = *dev_num;
	double a = *dev_param_a;
	double b = *dev_param_b;
	int i = threadIdx.x;
	for (int j = i + 1; j < num; j++) 
	{
		point * tmp_node = generate_intersection_point_gpu(&dev_line_array[i], &dev_line_array[j]);
		for (int k = 0; k < num; k++) 
		{
			double tmp = a * tmp_node->pos_x + b * tmp_node->pos_y;
			if ((tmp_node->pos_x * dev_line_array[k].param_a + tmp_node->pos_y * dev_line_array[k].param_b) < dev_line_array[k].param_c - 0.000001)
			{
				//printf("%f", tmp);
				break;
			}
			else if (k == num - 1)
			{
				valmax[i * num + j] = tmp;
			}
		}
	}
}

/*
int main() {
	// 1. get the input data
	inputs * input = read_from_file("C:/Users/hzauz/Desktop/sws3003_assignment 1/test_cases/100_0.dat");
	int num = input->number;
	printf("%d\n", num);
	// 2. transform data from CPU to GPU
	double * valmax = (double*)malloc(sizeof(double) * num * num);                      // for output
	for (int i = 0; i < num * num; i++)
	{
		valmax[i] = FLT_MAX;
	}
	double * dev_valmax;
	cudaMalloc((void**)&dev_valmax, sizeof(double) * num * num);
	cudaMemcpy(dev_valmax, valmax, sizeof(double) * num * num, cudaMemcpyHostToDevice);
	
	line * line_array = (line*)malloc(sizeof(line) * num);                              // for input line data
	for (int i = 0; i < num; i++)
	{
		line_array[i] = * ((input->lines)[i]);
	}
	line * dev_line_array;
	cudaMalloc((void**)&dev_line_array, sizeof(line) * num);
	cudaMemcpy(dev_line_array, line_array, sizeof(line) * num, cudaMemcpyHostToDevice);
	
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
	gpu_compute<<<1,num>>>(dev_num, dev_param_a, dev_param_b, dev_line_array, dev_valmax);

	// 4.getting the final answer
	cudaMemcpy(valmax,dev_valmax, sizeof(double*) * num * num, cudaMemcpyDeviceToHost);
	int minidx_x = 0;
	int minidx_y = 0;
	double minval = FLT_MAX;
	for (int i = 0; i < num * num; i++)
	{	
		if (valmax[i] < minval)
		{
			minidx_x = i / num;
			minidx_y = i - minidx_x * num;
			minval = valmax[i];
		}
	}
	printf("minidx_x:%d, minidy_y:%d", minidx_x, minidx_y);
	answer * ans = (answer *)malloc(sizeof(answer));
	point * tmp_node = generate_intersection_point(input->lines[minidx_x], input->lines[minidx_y]);
	double tmp = input->obj_function_param_a * tmp_node->pos_x + input->obj_function_param_b * tmp_node->pos_y;
	ans->answer_b = tmp;
	ans->intersection_point = tmp_node;
	ans->line1 = input->lines[minidx_x];
	ans->line2 = input->lines[minidx_y];
	// 3. display result and free memory
	char * ans_string = generate_ans_string(ans);
	printf("%s", ans_string);
	free_inputs(&input);
	cudaFree(dev_line_array);
	free(line_array);
	free_ans(&ans);
	free(ans_string);
	cudaFree(dev_valmax);
	free(valmax);

	return 0;
}
*/
struct rotate_functor
{
	const double a;
	const double b;

	rotate_functor(double _a, double _b) : a(_a), b(_b) {}

	__host__ __device__
		line operator() (line&rotate_line)
	{
		double x, y, z;
		x = rotate_line.param_a;
		y = rotate_line.param_b;
		z = rotate_line.param_c;
		bool big_0_before_rotate = z > 0;
		rotating(x, y, z);
		bool big_0_after_rotate = z > 0;
		if (big_0_before_rotate != big_0_after_rotate)
		{
			x = -x;
			y = -y;
			z = -z;
		}
		rotate_line.param_a = x;
		rotate_line.param_b = y;
		rotate_line.param_c = z;
		return rotate_line;
	}

	__host__ __device__
		void rotating(double& u, double& v, double& w)
	{
		double point_one_x, point_one_y, point_two_x, point_two_y = 0;
		point_one_x = (a * w) / u;
		point_one_y = (b * w) / u;
		point_two_x = (b * w) / v;
		point_two_y = ((-a) * w) / v;
		get_position(point_one_x, point_one_y);
		get_position(point_two_x, point_two_y);
		u = point_one_y - point_two_y;
		v = -(point_one_x - point_two_x);
		w = u * point_two_x + v * point_two_y;
	}

	__host__ __device__
		void get_position(double& pos_x, double& pos_y)
	{
		if (pos_x < 0)
		{
			pos_x = -(sqrt(abs(pos_x)));
		}
		else
			pos_x = sqrt(pos_x);
		if (pos_y < 0)
		{
			pos_y = -(sqrt(abs(pos_y)));
		}
		else
			pos_y = sqrt(pos_y);
	}

};

int main()
{
	inputs * input = read_from_file(""); //replace this with your input file
	int num = input->number;
	//line ** lines = input->lines;
	//thrust::device_vector<line> dev_lines(num);
	//for (int i = 0; i < num; i++)
	//{
	//	dev_lines[i] = (*lines)[i];
	//}
	for (int i = 0; i < num; i++)
	{
		std::cout << (*input->lines)[i].param_a << "\n" << std::endl;
	}
	double a, b;
	a = input->obj_function_param_a;
	b = input->obj_function_param_b;
	//thrust::transform(dev_lines.begin(), dev_lines.end(), dev_lines.begin(), rotate_functor(a, b));	
	//for (int i = 0; i < num; i++)
	//{
	//	std::cout << dev_lines[i].param_a << "\n" << std::endl;
	//}
	return 0;
}