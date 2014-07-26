
#include "da_svm.h"


void stoc_da_l1svm( ){
    double a;
    
    
}


int main(int argc, char **argv)
{
    printf("-----------------------\n");
    printf("-----scg_da_svm--------\n");
    printf("-----------------------\n");

    char input_file_name[1024];
    char model_file_name[1024];
    parse_command_line(argc, argv, input_file_name, model_file_name);

    load_data_file(input_file_name);

    train_online(model_file_name);

    libsvm_save_model(model_file_name);

}