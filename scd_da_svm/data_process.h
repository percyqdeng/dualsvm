/* 
 * File:   data_process.h
 * Author: qd
 *
 * Created on July 18, 2014, 4:12 PM
 */

#ifndef DATA_PROCESS_H
#define	DATA_PROCESS_H
#include <cstdio>

int libsvm_load_data(char *filename);

void load_data_file(char *filename);

void parse_command_line(int argc, char **argv, 
        char *input_file_name, char *test_file_name);





#endif	/* DATA_PROCESS_H */

