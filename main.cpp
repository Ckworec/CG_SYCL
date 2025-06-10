#include <stdio.h>
#include <math.h>
#include <thread>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sycl/sycl.hpp>
#include <mkl.h>
#define eps 0.01

auto nvidia_selector = [](const sycl::device& dev) {
    const std::string name = dev.get_info<sycl::info::device::name>();
    if (name.find("NVIDIA") != std::string::npos) {
        return 1;
    }
    return -1;
};

static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const& e : e_list) {
        try {
            std::rethrow_exception(e);
        }
        catch (std::exception const& e) {
#if _DEBUG
            std::cout << "Failure" << std::endl;
#endif
            std::terminate();
        }
    }
};

double scalar_product_parallel(sycl::queue& q, const std::vector<double>& a, const std::vector<double>& b)
{
    size_t n = a.size();
    double result = 0.0;

    for (int i = 0; i < n; ++i){
        result += a[i] * b[i];
    }
    
    return result;
}

void CG_SYCL(std::vector<double>& K_val, std::vector<size_t>& K_col_ind, std::vector<size_t>& K_row_ptr, std::vector<double>& x, std::vector<double>& b, std::string& device)
{
    sycl::queue cpu_queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
    sycl::queue gpu_queue(nvidia_selector, sycl::property::queue::in_order());
    sycl::queue q = (device == "CPU") ? cpu_queue : gpu_queue;
    bool is_gpu = (device == "GPU");

    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    double b_norm = scalar_product_parallel(q, b, b, is_gpu);

    if (b_norm < 1e-10)
    {
        std::fill(x.begin(), x.end(), 0.0);
        std::cout << "||b|| = 0" << std::endl;
        return;
    }

    size_t n = K_row_ptr.size();

    std::vector<double> r = b;
    std::vector<double> p = r;
    std::vector<double> Ap(n);

    unsigned int iteration = 0;
    unsigned int max_iter = K_row_ptr.size();

    sycl::buffer<double, 1> val_buf(K_val.data(), sycl::range<1>(K_val.size()));
    sycl::buffer<size_t, 1> col_ind_buf(K_col_ind.data(), sycl::range<1>(K_col_ind.size()));
    sycl::buffer<size_t, 1> row_ptr_buf(K_row_ptr.data(), sycl::range<1>(K_row_ptr.size()));
    sycl::buffer<double, 1> x_buf(x.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> r_buf(r.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> p_buf(p.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> Ap_buf(Ap.data(), sycl::range<1>(n));

    do {
        q.submit([&](sycl::handler& h) {
            sycl::accessor val_access(val_buf, h, sycl::read_only);
            sycl::accessor row_ptr_access(row_ptr_buf, h, sycl::read_only);
            sycl::accessor col_ind_access(col_ind_buf, h, sycl::read_only);
            sycl::accessor p_access(p_buf, h, sycl::read_only);
            sycl::accessor Ap_access(Ap_buf, h, sycl::write_only, sycl::no_init);
            h.parallel_for<class MatrixMul>(sycl::range<1>(n), [=](sycl::id<1> i) {
                double val(0.0);
                for (size_t k = row_ptr_access[i]; k < row_ptr_access[i + 1]; ++k) {
                    val += val_access[k] * p_access[col_ind_access[k]];
                }
                Ap_access[i] = val;
            });
        }).wait();

        // Copy Ap to host for scalar product
        q.submit([&](sycl::handler& h) {
            auto acc = Ap_buf.get_access<sycl::access::mode::read>(h);
            h.copy(acc, Ap.data());
        });
        q.wait_and_throw();

        double old_rr = scalar_product_parallel(q, r, r, is_gpu);
        double alpha = old_rr / scalar_product_parallel(q, p, Ap, is_gpu);

        q.submit([&](sycl::handler& h) {
            sycl::accessor x_access(x_buf, h, sycl::read_write);
            sycl::accessor r_access(r_buf, h, sycl::read_write);
            sycl::accessor p_access(p_buf, h, sycl::read_only);
            sycl::accessor Ap_access(Ap_buf, h, sycl::read_only);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                x_access[i] += alpha * p_access[i];
                r_access[i] -= alpha * Ap_access[i];
            });
        }).wait();

        // Copy r to host for scalar product
        q.submit([&](sycl::handler& h) {
            auto acc = r_buf.get_access<sycl::access::mode::read>(h);
            h.copy(acc, r.data());
        });
        q.wait_and_throw();

        double beta = scalar_product_parallel(q, r, r, is_gpu) / old_rr;

        q.submit([&](sycl::handler& h) {
            sycl::accessor p_access(p_buf, h, sycl::read_write);
            sycl::accessor r_access(r_buf, h, sycl::read_only);
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                p_access[i] = r_access[i] + beta * p_access[i];
            });
        }).wait();

        q.submit([&](sycl::handler& h) {
            auto acc = p_buf.get_access<sycl::access::mode::read>(h);
            h.copy(acc, p.data());
        });
        q.wait_and_throw();

        iteration++;
    } while (scalar_product_parallel(q, r, r, is_gpu) / b_norm > eps * eps && iteration < max_iter);

    q.submit([&](sycl::handler& h) {
        auto acc = x_buf.get_access<sycl::access::mode::read>(h);
        h.copy(acc, x.data());
    });
    q.wait_and_throw();

    std::cout << "iterations: " << iteration << std::endl;
}

int main(void){
    std::ifstream A_mat("A.txt");
    std::ifstream b_vec("B.vec");
    std::string CPU = "CPU", GPU = "GPU";
    size_t rows, nnz, len;
    struct timespec ts1, ts2;
    double sec = 0;

    A_mat >> rows >> nnz;
    b_vec >> len;

    std::vector<double> K_val(nnz);
    std::vector<size_t> K_row_ptr(rows);
    std::vector<size_t> K_col_ind(nnz);
    std::vector<double> b(len);

    for (size_t i = 0; i <= rows; ++i){
        A_mat >> K_row_ptr[i];
    }
    for (size_t i = 0; i < nnz; ++i){
        A_mat >> K_col_ind[i];
        b_vec >> b[i];
    }
    for (size_t i = 0; i < nnz; ++i){
        A_mat >> K_val[i];
    }

    A_mat.close();

    auto now = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(now);
    std::cout << "Время начала вычислений: " << std::ctime(&start_time);

	timespec_get(&ts1, TIME_UTC);
	std::vector<double> X1(b.size());
	CG_SYCL(K_val, K_col_ind, K_row_ptr, X1, b, CPU);
	timespec_get(&ts2, TIME_UTC);
	sec = (double(ts2.tv_sec) + double(ts2.tv_nsec) / 1000000000) - (double(ts1.tv_sec) + double(ts1.tv_nsec) / 1000000000);
	std::cout << "CPU: " << sec << std::endl << std::endl;

    now = std::chrono::system_clock::now();
    start_time = std::chrono::system_clock::to_time_t(now);
    std::cout << "Время начала вычислений: " << std::ctime(&start_time);

	timespec_get(&ts1, TIME_UTC);
	std::vector<double> X2(b.size());
	CG_SYCL(K_val, K_col_ind, K_row_ptr, X2, b, GPU);
	timespec_get(&ts2, TIME_UTC);
	sec = (double(ts2.tv_sec) + double(ts2.tv_nsec) / 1000000000) - (double(ts1.tv_sec) + double(ts1.tv_nsec) / 1000000000);
	std::cout << "GPU: " << sec << std::endl << std::endl;

    for (size_t i = 0; i < X1.size(); ++i){
        if (fabs(X1[i] - X2[i]) < eps * eps){
            std::cout << "Vse huinya na" << i << std::endl;
            return 0;
        }
    }

    std::cout << "Vse zaebis" << std::endl;

    return 0;
}