#include "func.hpp"

void read_vec(std::vector<double>& vec,
            std::string& name_file,
            size_t nnz)
{
    vec.resize(nnz);

    std::ifstream vec_file(name_file);

    for (size_t i = 0; i < nnz; ++i){
        vec_file >> vec[i];
    }

    vec_file.close();
}

void copy_from_nvidia(sycl::queue& q, sycl::buffer<double>& a_buf, double& a) {
    q.submit([&](sycl::handler& h) {
        sycl::accessor acc(a_buf, h, sycl::read_only);
        h.copy(acc, &a);
    }).wait_and_throw();
}

void copy_from_nvidia(sycl::queue& q, sycl::buffer<double>& a_buf, std::vector<double>& a) {
    q.submit([&](sycl::handler& h) {
        sycl::accessor acc(a_buf, h, sycl::read_only);
        h.copy(acc, a.data());
    }).wait_and_throw();
}

void test_CG_SYCL_jacobi(CSR_matrix<double>& mat, 
                        std::vector<double>& x, 
                        std::vector<double>& b, 
                        std::string& device)
{
    struct timespec ts1, ts2;
    double sec = 0;

    auto now = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(now);
    std::cout << "Время начала вычислений: " << std::ctime(&start_time);

    timespec_get(&ts1, TIME_UTC);
    CG_SYCL_jacobi(mat, x, b, device);
    timespec_get(&ts2, TIME_UTC);
    sec = (double(ts2.tv_sec) + double(ts2.tv_nsec) / 1000000000) - (double(ts1.tv_sec) + double(ts1.tv_nsec) / 1000000000);
    std::cout << device << " CG: " << sec << std::endl << std::endl;
}