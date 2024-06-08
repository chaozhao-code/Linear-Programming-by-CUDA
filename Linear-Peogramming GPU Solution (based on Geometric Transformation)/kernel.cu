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
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>

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
		line copy_line = rotate_line;
		x = copy_line.param_a;
		y = copy_line.param_b;
		z = copy_line.param_c;
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
		if (rotate_line.param_b < 0)
		{
			rotate_line.I_value = I_minus;
		}
		else if (rotate_line.param_b > 0)
		{
			rotate_line.I_value = I_plus;
		}
		else 
			rotate_line.I_value = I_0;
		rotate_line.slope_value = -(x / y);
		return rotate_line;
	}

	__host__ __device__
		void rotating(double& u, double& v, double& w)
	{
		double point_one_x, point_one_y, point_two_x, point_two_y = 0;
		point_one_y = (a * w) / u;
		point_one_x = (b * w) / u;
		point_two_y = (b * w) / v;
		point_two_x = ((-a) * w) / v;
		get_position(point_one_x, point_one_y);
		get_position(point_two_x, point_two_y);
		u = point_one_y - point_two_y;
		v = point_two_x - point_one_x;
		w = point_one_y * point_two_x - point_two_y * point_one_x; 
	}

	__host__ __device__
		void get_position(double& pos_x, double& pos_y)
	{
		double obj_length = sqrt(a*a + b*b);
		pos_x = pos_x / obj_length;
		pos_y = pos_y / obj_length;

	}

};

/*
struct I_functor
{
	const double a;

	I_functor(double _a) : a(_a) {}
	__host__ __device__
		I operator() (line&anyline)
	{
		if (anyline.param_b > 0)
			return I_pos;
		else if (anyline.param_b < 0)
			return I_neg;
		else
			return I_0;
	}
};
*/

struct generate_intersection
{
	const double x;
	generate_intersection(double _x) : x(_x) {}
	__host__ __device__
		line operator() (line&anyline)
	{
		line tmp_line = anyline;
		tmp_line.distance_with_testline = (tmp_line.param_c - tmp_line.param_a * x) / tmp_line.param_b;
		return tmp_line;
	}
};

void rotate_all(inputs*& input)
{
	line** lines = input->lines;
	int num = input->number;
	double pivot_a = input->obj_function_param_a;
	double pivot_b = input->obj_function_param_b;
	input->obj_function_param_a = 0;
	input->obj_function_param_b = sqrt(pivot_a * pivot_a + pivot_b * pivot_b);
	double cosine_value = pivot_b / (sqrt(pivot_a * pivot_a + pivot_b * pivot_b));
	double sine_value = pivot_a / (sqrt(pivot_a * pivot_a + pivot_b * pivot_b));
	for (int i = 0; i < num; ++i) {
		double new_a = (lines[i]->param_a * cosine_value - lines[i]->param_b * sine_value);
		double new_b = (lines[i]->param_a * sine_value + lines[i]->param_b * cosine_value);
		line* new_line = generate_line_from_abc(new_a, new_b, lines[i]->param_c);
		lines[i] = new_line;
	}
}

struct is_I_minus
{
	__host__ __device__
		bool operator()(line anyline)
	{
		return (anyline.I_value == -1);
	}
};

struct compare_by_distance
{
	__host__ __device__
		bool operator()(line line_one, line line_two)
	{
		return line_one.distance_with_testline < line_two.distance_with_testline;
	}
};

struct is_move
{
	const double slope;
	const point max_intersection;
	is_move(double _slope, point _max_intersection) : slope(_slope), max_intersection(_max_intersection) {}
	__host__ __device__
		bool operator()(line anyline)
	{
		if (anyline.slope_value + 0.0000001 < slope)   // + 0.001 ��Ϊ�˷�ֹ��������
			return true;
		else if (max_intersection.pos_y > (((anyline.param_c - anyline.param_a * max_intersection.pos_x) / anyline.param_b) + 0.0000001))
			return true;
		else
			return false;
	}
};

struct is_move_for_count
{
	const double slope;
	const point max_intersection;
	is_move_for_count(double _slope, point _max_intersection) : slope(_slope), max_intersection(_max_intersection) {}
	__host__ __device__
		bool operator()(line anyline)
	{
		if (anyline.slope_value + 0.0000001 < slope)   // + 0.001 ��Ϊ�˷�ֹ��������
			return false;
		else if (max_intersection.pos_y > (((anyline.param_c - anyline.param_a * max_intersection.pos_x) / anyline.param_b) + 0.0000001))
			return false;
		else
			return true;
	}
};

struct fanzao
{
	const double x;
	fanzao(double _x) : x(_x) {}
	__host__ __device__
		double operator()(line anyline)
	{
		return (anyline.param_c - anyline.param_a * x) / anyline.param_b;
	}
};

struct one_plus_find_answer
{
	const line plus_line;
	one_plus_find_answer(line _plus_line) : plus_line(_plus_line) {}
	__host__ __device__
		double operator()(line minus_line)
	{
		return (plus_line.param_c * minus_line.param_a - plus_line.param_a * minus_line.param_c) / (plus_line.param_b * minus_line.param_a - plus_line.param_a * minus_line.param_b);
	}

};

void print_line(line * anyline)
{	
	std::cout << anyline->param_a << "x" << "+" << anyline->param_b << "y" << ">=" << anyline->param_c;
	std::cout << "   I_value" << anyline->I_value << "  slope" << anyline->slope_value << "  distance" << anyline->distance_with_testline;
	std::cout << std::endl;
}

void print_answer_by_me(answer_me ans)
{
	std::cout << "Answer is: " << ans.answer_b << std::endl;
	std::cout << "Line 1 is: " << ans.line1.param_a << "x +" << ans.line1.param_b << "y >=" << ans.line1.param_c << std::endl;
	std::cout << "Line 2 is: " << ans.line2.param_a << "x +" << ans.line2.param_b << "y >=" << ans.line2.param_c << std::endl;
	std::cout << "Intersection pointer is posX : " << ans.intersection_point.pos_x << ",   posY : " << ans.intersection_point.pos_y << std::endl;
}

void intersection_rotate_return(point &anypoint, double a, double b)
{
	double x = anypoint.pos_x;
	double y = anypoint.pos_y;
	double object_length = sqrt(a * a + b * b);
	anypoint.pos_x = (x * b + y * a) / object_length;
	anypoint.pos_y = (x * (-a) + y * b) / object_length;
}

void line_rotate_return(line &anyline, double a, double b)
{
	point node_x, node_y;
	node_x.pos_y = 0;
	node_x.pos_x = anyline.param_c / anyline.param_a;
	node_y.pos_x = 0;
	node_y.pos_y = anyline.param_c / anyline.param_b;
	intersection_rotate_return(node_x, a, b);
	intersection_rotate_return(node_y, a, b);
	bool z_0_befor = anyline.param_c > 0;
	anyline = *(generate_line_from_2points(&node_x, &node_y));
	bool z_0_after = anyline.param_c > 0;
	if (z_0_befor != z_0_after)
	{
		anyline.param_a = -anyline.param_a;
		anyline.param_b = -anyline.param_b;
		anyline.param_c = -anyline.param_c;
	}
}

void answer_rotate(answer_me&ans, double a, double b)
{
	line_rotate_return(ans.line1, a, b);
	line_rotate_return(ans.line2, a, b);
	intersection_rotate_return(ans.intersection_point, a, b);
}

int main()
{
	inputs * input = read_from_file(""); // replace this with your input files
	int num = input->number;
	line ** lines = input->lines;
	/*
	for (int i = 0; i < num; i++)
	{
		dev_lines.push_back(* lines[i]);
	}
	
	thrust::host_vector<line> host_lines(num);
	for (int i = 0; i < num; i++)
	{
		host_lines[i] = (*lines[i]);
	}
	*/
	thrust::device_vector<line> device_lines(num);
	for (int i = 0; i < num; i++)
	{
		device_lines[i] = (*lines[i]);
	}
	/*
	thrust::host_vector<line> host_lines(num);
	thrust::copy(device_lines.begin(), device_lines.end(), host_lines.begin());
	for (int i = 0; i < num; i++)
	{
		std::cout << host_lines[i].param_a << "\n" << std::endl;
	}
	*/
	double a, b;
	a = input->obj_function_param_a;
	b = input->obj_function_param_b;
	double object_length = sqrt(a*a + b*b);

	thrust::transform(device_lines.begin(), device_lines.end(), device_lines.begin(), rotate_functor(a, b));	
	thrust::host_vector<line> host_lines(num);
	/*
	thrust::copy(device_lines.begin(), device_lines.end(), host_lines.begin());
	for (int i = 0; i < num; i++)
	{
		std::cout << i << "   " << host_lines[i].param_a;
		std::cout << "   " << host_lines[i].param_b;
		std::cout << "   " << host_lines[i].param_c;
		std::cout << "   " << host_lines[i].I_value;
		std::cout << "\n" << std::endl;
	}
	*/


	// 7��26��ע��
	/*thrust::transform(device_lines.begin(), device_lines.end(), device_lines.begin(), generate_intersection(0));
	
	thrust::copy(device_lines.begin(), device_lines.end(), host_lines.begin());
	for (int i = 0; i < num; i++)
	{
		std::cout << i << "   " << host_lines[i].param_a;
		std::cout << "   " << host_lines[i].param_b;
		std::cout << "   " << host_lines[i].param_c;
		std::cout << "   " << host_lines[i].I_value;
		std::cout << "   " << host_lines[i].slope_value;
		std::cout << "   " << host_lines[i].distance_with_testline;
		std::cout << "\n" << std::endl;
	}*/



	//thrust::host_vector<intersection> host_lines_insertion(num);
	//thrust::copy(device_lines_insertion.begin(), device_lines_insertion.end(), host_lines_insertion.begin());

	//for (int i = 0; i < num; i++)
	//{
	//	std::cout << i << "   " << host_lines_insertion[i].pos_x;
	//	std::cout << "   " << host_lines_insertion[i].pos_y;
	//	std::cout << "\n" << std::endl;
	//}
	

	
	int number_of_Iminus = thrust::count_if(device_lines.begin(), device_lines.end(), is_I_minus());
	int number_of_Iplus = num - number_of_Iminus;
	//std::cout << number_of_Iminus << std::endl;
	thrust::device_vector<line> device_I_minus_lines(number_of_Iminus);
	thrust::device_vector<line> device_I_plus_lines(num - number_of_Iminus);
	thrust::partition_copy(device_lines.begin(), device_lines.end(), device_I_minus_lines.begin(), device_I_plus_lines.begin(), is_I_minus());
	//std::cout << device_I_minus_lines.size() << std::endl;
	//std::cout << device_I_plus_lines.size() << std::endl;
	answer_me ans;
	if (number_of_Iplus == 1)
	{
		line Iplus = device_I_plus_lines[0];
		thrust::device_vector<double> device_potential_answer(number_of_Iminus);
		thrust::host_vector<double> host_potential_answer(number_of_Iminus);
		thrust::transform(device_I_minus_lines.begin(), device_I_minus_lines.end(), device_potential_answer.begin(), one_plus_find_answer(Iplus));
		thrust::copy(device_potential_answer.begin(), device_potential_answer.end(), host_potential_answer.begin());
		double min = host_potential_answer[0];
		int minidx = 0;
		for (int i = 0; i < number_of_Iminus; i++)
		{
			if (host_potential_answer[i] < min)
			{
				min = host_potential_answer[i];
				minidx = i;
			}
		}
		ans.answer_b = min * object_length;
		ans.line1 = Iplus;
		ans.line2 = device_I_minus_lines[minidx];
		ans.intersection_point.pos_x = (Iplus.param_c * ans.line2.param_b - Iplus.param_b * ans.line2.param_c)
			/ (Iplus.param_a * ans.line2.param_b - Iplus.param_b * ans.line2.param_a);
		ans.intersection_point.pos_y = (Iplus.param_c * ans.line2.param_a - Iplus.param_a * ans.line2.param_c)
			/ (Iplus.param_b * ans.line2.param_a - Iplus.param_a * ans.line2.param_b);
	}
	if (number_of_Iplus > 2)
	{
		// ����Ϊɾ��I+ֱ�ߵĲ���

		double testline = 50;
		// ѡȡһ����ֱ��ֱ����Ϊ�ұ߽�,���ҵ��ұ߽��ϵ�����I+ֱ��
		line max_I_plus_line_rightbound;  // ���ұ߽�����I+ֱ��
		double rightbound;                    // �ұ߽�
		line max_I_plus_line;             // ����߽�����I+ֱ��
		double leftbound;                     // ��߽�

											  // ȷ��һ���ұ߽�
		while (true)
		{
			thrust::transform(device_I_plus_lines.begin(), device_I_plus_lines.end(), device_I_plus_lines.begin(), generate_intersection(testline));
			auto o_0 = thrust::max_element(device_I_plus_lines.begin(), device_I_plus_lines.end(), compare_by_distance());
			max_I_plus_line_rightbound = device_I_plus_lines[o_0 - device_I_plus_lines.begin()];
			//std::cout << max_I_minus_line_rightbound.param_a << "  " << max_I_minus_line_rightbound.slope_value << std::endl;
			/*thrust::host_vector<line> host_I_minus_lines(number_of_Iminus);
			thrust::copy(device_I_minus_lines.begin(), device_I_minus_lines.end(), host_I_minus_lines.begin());*/
			/*for (int i = 0; i < number_of_Iminus; i++)
			{
			std::cout << i << "   " << host_I_minus_lines[i].param_a;
			std::cout << "   " << host_I_minus_lines[i].param_b;
			std::cout << "   " << host_I_minus_lines[i].param_c;
			std::cout << "   " << host_I_minus_lines[i].I_value;
			std::cout << "   " << host_I_minus_lines[i].slope_value;
			std::cout << "   " << host_I_minus_lines[i].distance_with_testline;
			std::cout << "\n" << std::endl;
			}
			break;*/
			if (max_I_plus_line_rightbound.slope_value > 0)
				break;
				//std::cout << max_I_plus_line_rightbound.slope_value << std::endl;
			testline = testline + 10;  // �����ֱ��Ϊ��߽磬������
		}
		rightbound = testline;
		//std::cout << "the rightbound is " << rightbound << std::endl;
		//print_line(&max_I_plus_line_rightbound);

		// �ұ߽�����ѡȡһ��ֱ�ߣ�������߽磬������I+ֱ��б��С��0����б�ʴ���0�����Ը�ֱ��Ϊ�ұ߽磬����Ѱ����߽�
		while (true)
		{
			testline = rightbound - 10;
			thrust::transform(device_I_plus_lines.begin(), device_I_plus_lines.end(), device_I_plus_lines.begin(), generate_intersection(testline));
			auto o_0 = thrust::max_element(device_I_plus_lines.begin(), device_I_plus_lines.end(), compare_by_distance());
			max_I_plus_line = device_I_plus_lines[o_0 - device_I_plus_lines.begin()];
			//std::cout << max_I_minus_line.distance_with_testline << std::endl;
			if (max_I_plus_line.slope_value > 0)
			{
				rightbound = testline;
				//std::cout << "now the rightbound is " << rightbound << std::endl;
				max_I_plus_line_rightbound = max_I_plus_line;
				continue;
			}
			else
				leftbound = testline;
			break;
		}
		//std::cout << "the leftbound is " << leftbound << std::endl;
		//print_line(&max_I_plus_line);
		line for_move_bound;
		while (device_I_plus_lines.size() > 2)
		{
			// ������߽��ϵ����I+ֱ��
			thrust::transform(device_I_plus_lines.begin(), device_I_plus_lines.end(), device_I_plus_lines.begin(), generate_intersection(leftbound));
			auto o_0 = thrust::max_element(device_I_plus_lines.begin(), device_I_plus_lines.end(), compare_by_distance());
			max_I_plus_line = device_I_plus_lines[o_0 - device_I_plus_lines.begin()];
			//print_line(&max_I_plus_line);

			/*thrust::host_vector<line> host_I_plus_lines(device_I_plus_lines.size());
			thrust::copy(device_I_plus_lines.begin(), device_I_plus_lines.end(), host_I_plus_lines.begin());
			std::cout << "the size of I plus line" << device_I_plus_lines.size() << std::endl;
			for (int i = 0; i < device_I_plus_lines.size(); i++)
			{
				std::cout << i << "   " << host_I_plus_lines[i].param_a;
				std::cout << "   " << host_I_plus_lines[i].param_b;
				std::cout << "   " << host_I_plus_lines[i].param_c;
				std::cout << "   " << host_I_plus_lines[i].I_value;
				std::cout << "   " << host_I_plus_lines[i].slope_value;
				std::cout << "   " << host_I_plus_lines[i].distance_with_testline;
				std::cout << "\n" << std::endl;
			}*/
			// �����ұ߽��ϵ����I+ֱ��
			thrust::transform(device_I_plus_lines.begin(), device_I_plus_lines.end(), device_I_plus_lines.begin(), generate_intersection(rightbound));
			o_0 = thrust::max_element(device_I_plus_lines.begin(), device_I_plus_lines.end(), compare_by_distance());
			max_I_plus_line_rightbound = device_I_plus_lines[o_0 - device_I_plus_lines.begin()];
			//print_line(&max_I_plus_line_rightbound);

			////thrust::host_vector<line> host_I_plus_lines(device_I_plus_lines.size());
			//thrust::copy(device_I_plus_lines.begin(), device_I_plus_lines.end(), host_I_plus_lines.begin());
			//std::cout << "the size of I plus line" << device_I_plus_lines.size() << std::endl;
			//for (int i = 0; i < device_I_plus_lines.size(); i++)
			//{
			//	std::cout << i << "   " << host_I_plus_lines[i].param_a;
			//	std::cout << "   " << host_I_plus_lines[i].param_b;
			//	std::cout << "   " << host_I_plus_lines[i].param_c;
			//	std::cout << "   " << host_I_plus_lines[i].I_value;
			//	std::cout << "   " << host_I_plus_lines[i].slope_value;
			//	std::cout << "   " << host_I_plus_lines[i].distance_with_testline;
			//	std::cout << "\n" << std::endl;
			//}



			// �Ƴ�ֱ���Լ��ж����ұ߽����I+ֱ�ߵĽ��㴦�Ĵ�������߽绹���ұ߽�
			point * max_intersection = generate_intersection_point(&max_I_plus_line, &max_I_plus_line_rightbound);
			/*if (device_I_plus_lines.size() == 4)
			{
				std::cout << "intersection" << "(" << max_intersection->pos_x << "," << max_intersection->pos_y << std::endl;

				thrust::device_vector<double> device_fanzao(device_I_plus_lines.size());
				thrust::host_vector<double> host_fanzao(device_I_plus_lines.size());
				thrust::transform(device_I_plus_lines.begin(), device_I_plus_lines.end(), device_fanzao.begin(), fanzao(max_intersection->pos_x));
				thrust::copy(device_fanzao.begin(), device_fanzao.end(), host_fanzao.begin());

				for (int i = 0; i < device_I_plus_lines.size(); i++)
				{
				std::cout << i << " " << host_fanzao[i] << std::endl;
				}
			}*/
			/*std::cout << "intersection" << "(" << max_intersection->pos_x << "," << max_intersection->pos_y << std::endl;

			thrust::device_vector<double> device_fanzao(device_I_plus_lines.size());
			thrust::host_vector<double> host_fanzao(device_I_plus_lines.size());
			thrust::transform(device_I_plus_lines.begin(), device_I_plus_lines.end(), device_fanzao.begin(), fanzao(max_intersection->pos_x));
			thrust::copy(device_fanzao.begin(), device_fanzao.end(), host_fanzao.begin());

			for (int i = 0; i < device_I_plus_lines.size(); i++)
			{
				std::cout << i << " " << host_fanzao[i] << std::endl;
			}*/


			int no_del = thrust::count_if(device_I_plus_lines.begin(), device_I_plus_lines.end(), is_move_for_count(max_I_plus_line.slope_value, *max_intersection));
			thrust::remove_if(device_I_plus_lines.begin(), device_I_plus_lines.end(), is_move(max_I_plus_line.slope_value, *max_intersection));
			//std::cout << "after remove" << device_I_plus_lines.size() << std::endl;
			//std::cout << "no_del   " << no_del << std::endl;
			device_I_plus_lines.resize(no_del);
			//std::cout << "device_I_plus_lines" << device_I_plus_lines.size() << std::endl;
			testline = max_intersection->pos_x;
			thrust::transform(device_I_plus_lines.begin(), device_I_plus_lines.end(), device_I_plus_lines.begin(), generate_intersection(testline));
			o_0 = thrust::max_element(device_I_plus_lines.begin(), device_I_plus_lines.end(), compare_by_distance());
			for_move_bound = device_I_plus_lines[o_0 - device_I_plus_lines.begin()];
			if (for_move_bound.slope_value > 0)
				rightbound = testline;
			else
				leftbound = testline;
			////std::cout << "now the rightbound is " << rightbound << std::endl;
			//thrust::host_vector<line> host_I_plus_lines(device_I_plus_lines.size());
			//thrust::copy(device_I_plus_lines.begin(), device_I_plus_lines.end(), host_I_plus_lines.begin());
			//std::cout << "the size of I plus line" << device_I_plus_lines.size() << std::endl;
			//for (int i = 0; i < device_I_plus_lines.size(); i++)
			//{
			//	std::cout << i << "   " << host_I_plus_lines[i].param_a;
			//	std::cout << "   " << host_I_plus_lines[i].param_b;
			//	std::cout << "   " << host_I_plus_lines[i].param_c;
			//	std::cout << "   " << host_I_plus_lines[i].I_value;
			//	std::cout << "   " << host_I_plus_lines[i].distance_with_testline;
			//	std::cout << "\n" << std::endl;
			//}
		}
		number_of_Iplus = device_I_plus_lines.size();
	}
	if (number_of_Iplus == 2)
	{
		ans.line1 = device_I_plus_lines[0];
		ans.line2 = device_I_plus_lines[1];
		point * line1_line2 = generate_intersection_point(&ans.line1, &ans.line2);
		ans.answer_b = line1_line2->pos_y * object_length;
		ans.intersection_point = *line1_line2;
	}
	print_answer_by_me(ans);
	answer_rotate(ans, a, b);
	print_answer_by_me(ans);


	//thrust::host_vector<line> host_I_minus_lines(device_I_minus_lines.size());
	//thrust::copy(device_I_minus_lines.begin(), device_I_minus_lines.end(), host_I_minus_lines.begin());
	//for (int i = 0; i < device_I_minus_lines.size(); i++)
	//{
	//	std::cout << i << "   " << host_I_minus_lines[i].param_a;
	//	std::cout << "   " << host_I_minus_lines[i].param_b;
	//	std::cout << "   " << host_I_minus_lines[i].param_c;
	//	std::cout << "   " << host_I_minus_lines[i].I_value;
	//	std::cout << "   " << host_I_minus_lines[i].distance_with_testline;
	//	std::cout << "\n" << std::endl;
	//}
	
	


	/*
	thrust::device_vector<I> I_of_eachline(num);
	thrust::transform(device_lines.begin(), device_lines.end(), I_of_eachline.begin(), I_functor(0));
	thrust::host_vector<I> host_I_of_lines(num);
	thrust::copy(I_of_eachline.begin(), I_of_eachline.end(), host_I_of_lines.begin());
	*/
	/*
	for (int i = 0; i < num; i++)
	{
		std::cout << i << "   " << host_lines[i].param_a;
		std::cout << i << "   " << host_lines[i].param_b;
		std::cout << i << "   " << host_lines[i].param_c;
		std::cout << i << "   " << host_I_of_lines[i];
		std::cout << "\n" << std::endl;
	}
	*/
	//for (int i = 0; i < num; i++)
	//{
		//(*lines[i]) = host_lines[i];
		//std::cout << i << "   " << (*lines[i]).param_a << std::endl;
	//}
	//inputs * rotate_input = (inputs *)malloc(sizeof(inputs));;
	//rotate_input->lines = lines;
	//for (int i = 0; i < num; i++)
	//{
		//(*lines[i]) = host_lines[i];
	//	std::cout << i << "   " << (*rotate_input->lines[i]).param_a << std::endl;
	//}
	//rotate_input->obj_function_param_a = 0;
	//rotate_input->obj_function_param_b = sqrt(a*a+b*b);
	/*
	answer * ans = compute(input);
	rotate_all(input);
	for (int i = 0; i < num; i++)
	{
		std::cout << i << "   " << (*input->lines[i]).param_a;
		std::cout << i << "   " << (*input->lines[i]).param_b;
		std::cout << i << "   " << (*input->lines[i]).param_c;
		std::cout << "\n" << std::endl;
	}
	char * ans_string = generate_ans_string(ans);
	printf("%s", ans_string);
	free_inputs(&input);
	//free_inputs(&rotate_input);
	//free_inputs(&rotate_input);
	free_ans(&ans);
	free(ans_string);
	*/
	return 0;
}