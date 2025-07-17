#include "func.hpp"

int main(void){
    std::cout << "Enter number file: ";
    std::string number;
    std::cin >> number;

    std::string prefix = "mat_vec/";
    std::string A_name = prefix + "A" + number + ".txt";
    std::string B_name = prefix + "B" + number + ".vec";
    std::string CPU = "CPU", GPU = "GPU";

    size_t rows, nnz, len;
    struct timespec ts1, ts2;
    double sec = 0;
    double average = 0.0;

    std::vector<double> b;
    CSR_matrix<double> mat;

    mat.read_mat(A_name);
    read_vec(b, B_name, mat.take_nnz());

    std::vector<double> X1(b.size());
    std::vector<double> X2(b.size());

    std::cout << "Matrix size: " <<  mat.take_rows() << std::endl << std::endl;

    test_CG_SYCL_jacobi(mat, X1, b, CPU);
    test_CG_SYCL_jacobi(mat, X2, b, GPU);

    for (size_t i = 0; i < X1.size(); ++i){
        average += fabs(X1[i] - X2[i]);
    }

    std::cout << "Average error CPU and GPU: " << average / X1.size() << std::endl;

    return 0;
}