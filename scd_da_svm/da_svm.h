/* 
 * File:   da_svm.h
 * Author: qd
 *
 * Created on July 18, 2014, 2:11 PM
 */

#ifndef DA_SVM_H
#define	DA_SVM_H
#include "kernel_matrix.h"

enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED };
class svm_parameters{
public:
    int kernel_type;
    
    double gamma;
    double coef0;
    double lmda;
    double * alpha;  /** weights of support vectors*/
    
    
};

class svm_model{
    svm_parameters para;
    int n;  /** size of training set*/
    
};





#endif	/* DA_SVM_H */

    