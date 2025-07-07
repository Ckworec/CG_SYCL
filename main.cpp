#include "include.hpp"

int main(void){
    std::ifstream A_mat("A.txt");
    std::ifstream b_vec("B.vec");
    std::string CPU = "CPU", GPU = "GPU";
    size_t rows, nnz, len;
    struct timespec ts1, ts2;
    double sec = 0;

    A_mat >> rows >> nnz;
    b_vec >> len;

    std::cout << "Matrix size: " << rows << std::endl << std::endl;

    std::vector<double> K_val(nnz);
    std::vector<size_t> K_row_ptr(rows + 1);
    std::vector<size_t> K_col_ind(nnz);
    std::vector<double> b(len);

    read(K_row_ptr, K_col_ind, K_val, b, b_vec, A_mat, nnz, rows);

    A_mat.close();
    b_vec.close();

    auto now = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(now);
    std::cout << "Время начала вычислений: " << std::ctime(&start_time);

    timespec_get(&ts1, TIME_UTC);
    std::vector<double> X1(b.size());
    CG_SYCL_jacobi(K_val, K_col_ind, K_row_ptr, X1, b, CPU);
    timespec_get(&ts2, TIME_UTC);
    sec = (double(ts2.tv_sec) + double(ts2.tv_nsec) / 1000000000) - (double(ts1.tv_sec) + double(ts1.tv_nsec) / 1000000000);
    std::cout << "CPU: " << sec << std::endl << std::endl;

    now = std::chrono::system_clock::now();
    start_time = std::chrono::system_clock::to_time_t(now);
    std::cout << "Время начала вычислений: " << std::ctime(&start_time);

    timespec_get(&ts1, TIME_UTC);
    std::vector<double> X2(b.size());
    CG_SYCL_jacobi(K_val, K_col_ind, K_row_ptr, X2, b, GPU);
    timespec_get(&ts2, TIME_UTC);
    sec = (double(ts2.tv_sec) + double(ts2.tv_nsec) / 1000000000) - (double(ts1.tv_sec) + double(ts1.tv_nsec) / 1000000000);
    std::cout << "GPU: " << sec << std::endl << std::endl;

    for (size_t i = 0; i < X1.size(); ++i){
        if (fabs(X1[i] - X2[i]) > eps * eps){
            std::cout << "Vse ploho na " << i << " and " << fabs(X1[i] - X2[i]) << " > " << eps * eps << std::endl;
            return 0;
        }
    }

    std::cout << "Vse prekrasno" << std::endl;

    return 0;
}