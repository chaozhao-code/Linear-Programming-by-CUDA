#include "floating_number_helper.h"
#include "input_output.h"

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
    answer * ans = (answer *) malloc(sizeof(answer));
    double tmp = 0;
    double min = MAXFLOAT;
    for (int i = 0; i < num - 1; i++) {
        line * now_line = lines[i];
        for (int j = i + 1; j < num; j++) {
            line * tmp_line = lines[j];
            point * tmp_node = generate_intersection_point(now_line, tmp_line);
            tmp = input->obj_function_param_a * tmp_node->pos_x + input->obj_function_param_b * tmp_node->pos_y;
            if (tmp <= min) {
                printf("%f",tmp);
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

int main() {
    // 1. get the input data
    inputs * input = read_from_file("C:/Users/hzauz/Desktop/sws3003_assignment 1/test_cases/1_0.dat");
    // 2. get the answer
    answer * ans = compute(input);
    // 3. display result and free memory
    char * ans_string = generate_ans_string(ans);
    printf("%s", ans_string);
    free_inputs(&input);
    free_ans(&ans);
    free(ans_string);
    return 0;
}