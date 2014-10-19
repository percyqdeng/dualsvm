#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>

using namespace std;

int main(){
	int n = 8000;
	double **arr;
	arr = new double*[n];
	int i, j;
	for(i=0; i<n; i++){
		arr[i] = new double[n];
	}
	clock_t start;
	start = clock();
	for(i=0; i<n; i++){
		for(j=0; j<n; j++){
			arr[i][j] = i+j;
		}
	}
	double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout<<"row first time cost "<<duration<<endl;

	start = clock();
	for(j=0; j<n; j++){
		for(i=0; i<n; i++){
			arr[i][j] = i+j;
		}
	}
	duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout<<"column first time cost "<<duration<<endl;
	return 0;
}
