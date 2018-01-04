#include <armadillo>
#include <iostream>
#include <string>

using namespace std;
using namespace arma;

int main(void)
{
	mat M = { 1,2,3,
		  4,5,6,
		  7,8,9 };
	M.reshape(3,3);

	mat M2, M3;

	ofstream mat1("mat1");

	M.save(mat1, arma_ascii);
	M.save(mat1, arma_ascii);
	
	mat1.close();

	ifstream matf("mat1");

	M2.load(matf, arma_ascii);
	M3.load(matf, arma_ascii);

	cout << M2 << M3;

}
