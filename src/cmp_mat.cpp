#include <iostream>
#include <cstdio>
#include <cstdlib>
using namespace std;

int main(){
	int n = 50;
	double arr[50][50];
	int i, j;
	for(i=0; i<n; i++){
		for(j=0;j<50; j++)
		arr[i][j] = rand()%10;
	}
	return 0;
}
