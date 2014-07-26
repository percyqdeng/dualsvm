/* 
 * File:   kernel_matrix.h
 * Author: qdengpercy
 *
 * Created on July 18, 2014, 9:46 AM
 */

#ifndef KERNEL_MATRIX_H

#define	KERNEL_MATRIX_H

class simple_kernel{
public:
    int n;
    double ** kmatrix;
    simple_kernel();
    simple_kernel(const simple_kernel& orig);
    
    virtual ~simple_kernel();
private:
    
};

#endif	/* KERNEL_MATRIX_H */

