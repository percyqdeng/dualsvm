/* 
 * File:   feature_vector.h
 * Author: qd
 *
 * Created on July 18, 2014, 3:31 PM
 */

#ifndef FEATURE_VECTOR_H
#define	FEATURE_VECTOR_H

struct feature_node{
    int index;
    double value;
};
class feature_vector{
public:
    feature_node * my_vec;
    feature_vector();
    feature_vector(const feature_vector& orig);
    virtual ~feature_vector();
private:

};

#endif	/* FEATURE_VECTOR_H */

