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

	sycl::buffer<double, 1> a_buf(a.data(), sycl::range<1>(n));
	sycl::buffer<double, 1> b_buf(b.data(), sycl::range<1>(n));

	double result = 0.0;
    {
	    sycl::buffer<double, 1> result_buf(&result, sycl::range<1>(1));

        q.submit([&](sycl::handler& h) {
            auto a_acc = a_buf.get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
            auto sum_reduction = sycl::reduction(result_buf, h, std::plus<double>());

            h.parallel_for(sycl::range<1>(n), sum_reduction, [=](sycl::id<1> i, auto& sum) {
                sum += a_acc[i] * b_acc[i];
            });
        }).wait();
    }

	return result;
}


void CG_SYCL(std::vector<double>& K_val, std::vector<size_t>& K_col_ind, std::vector<size_t>& K_row_ptr, std::vector<double>& x, std::vector<double>& b, std::string& device)
{
    sycl::queue cpu_queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
    sycl::queue gpu_queue(nvidia_selector, sycl::property::queue::in_order());
	sycl::queue q = (device == "CPU") ? cpu_queue : gpu_queue;

    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

	double b_norm = scalar_product_parallel(q, b, b);

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

	do
	{
		{
			sycl::buffer<double, 1> p_buf(p.data(), sycl::range<1>(n));
			sycl::buffer<double, 1> Ap_buf(Ap.data(), sycl::range<1>(n));

			q.submit([&](sycl::handler& h) {
				sycl::accessor val_access(val_buf, h, sycl::read_only);
				sycl::accessor row_ptr_access(row_ptr_buf, h, sycl::read_only);
				sycl::accessor col_ind_access(col_ind_buf, h, sycl::read_only);
				sycl::accessor p_access(p_buf, h, sycl::read_only);
				sycl::accessor Ap_access(Ap_buf, h, sycl::write_only, sycl::no_init);
				h.parallel_for<class MatrixMul>(sycl::range<1>(n), [=](sycl::id<1> i) {
					double val(0.0);
					for (size_t k = row_ptr_access[i]; k < row_ptr_access[i + 1]; ++k)
					{
						val += val_access[k] * p_access[col_ind_access[k]];
					}
					Ap_access[i] = val;
				});
			});
		}q.wait();
		
		double old_rr = scalar_product_parallel(q, r, r);
		double alpha = old_rr / scalar_product_parallel(q, p, Ap);

		for (size_t i = 0; i < n; ++i) 
		{
			x[i] += alpha * p[i];
			r[i] -= alpha * Ap[i];
		}

		double beta = scalar_product_parallel(q, r, r) / old_rr;
		
		for (size_t i = 0; i < n; ++i) 
		{
			p[i] = r[i] + beta * p[i];
		}
		
		iteration++;
	} while (scalar_product_parallel(q, r, r) / b_norm > eps * eps && iteration < max_iter);

	std::cout << "iterations: " << iteration << std::endl;
}

void solve_cg_mkl(const std::vector<size_t>& row_ptr, const std::vector<size_t>& col_idx, const std::vector<double>& values, const std::vector<double>& b, std::vector<double>& x)
{
    int n = static_cast<int>(b.size());

    std::vector<MKL_INT> row_ptr_int(row_ptr.begin(), row_ptr.end());
    std::vector<MKL_INT> col_idx_int(col_idx.begin(), col_idx.end());

    int ipar[128] = {};
    double dpar[128] = {};
    std::vector<double> tmp(4 * n);
    MKL_INT RCI_request;
    double tol = 0.0001;
    int max_iter = row_ptr.size();

    dcg_init(&n, x.data(), b.data(), &RCI_request, ipar, dpar, tmp.data());

    ipar[0] = 1;
    ipar[1] = 6;
    ipar[4] = max_iter;
    ipar[7] = 1;
    dpar[0] = tol;

    do {
        dcg(&n, x.data(), b.data(), &RCI_request, ipar, dpar, tmp.data());

        if (RCI_request == 0) {
            break;
        } else if (RCI_request == 1) {
            char matdescra[6] = {'G', '*', '*', 'C', '*', '\0'};
            double alpha = dpar[1];
            double beta = dpar[2];

            mkl_dcsrmv("N", &n, &n,
                       &alpha, matdescra,
                       values.data(),
                       col_idx_int.data(),
                       row_ptr_int.data(),
                       row_ptr_int.data() + 1,
                       tmp.data(),
                       &beta,
                       tmp.data() + n);
        } else {
            std::cerr << "Unsupported RCI_request = " << RCI_request << std::endl;
            break;
        }
    } while (true);

    std::cout << "Решение завершено за " << ipar[4] << " итераций." << std::endl;
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

    timespec_get(&ts1, TIME_UTC);
	std::vector<double> X5(b.size());
	solve_cg_mkl(K_row_ptr, K_col_ind, K_val, b, X5);
	timespec_get(&ts2, TIME_UTC);
	sec = (double(ts2.tv_sec) + double(ts2.tv_nsec) / 1000000000) - (double(ts1.tv_sec) + double(ts1.tv_nsec) / 1000000000);
	std::cout << "MKL: " << sec << std::endl << std::endl;

    for (size_t i = 0; i < X1.size(); ++i){
        if (fabs(X1[i] - X2[i]) < eps * eps && fabs(X2[i] - X5[i]) < eps * eps){
            std::cout << "Vse huinya na" << i << std::endl;
            return 0;
        }
    }

    std::cout << "Vse zaebis" << std::endl;

    return 0;
}