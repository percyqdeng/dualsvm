
#include <stdlib.h>
#include <iostream>

struct fruit{
    int size;
    fruit(){size = 1;}
};
int main(){
    fruit apple;
    printf("fruit size %d", apple.size);
    return 0;
}