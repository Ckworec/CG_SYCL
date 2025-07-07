#include <stdio.h>
#include <math.h>
#include <thread>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sycl/sycl.hpp>
#define eps 1.e-4

// --------------- Help functions ---------------------

void read(std::vector<size_t>& row_ptr, 
        std::vector<size_t>& col_ind, 
        std::vector<double>& val, 
        std::vector<double>& vec,
        std::ifstream& vec_file,
        std::ifstream& mat_file, 
        size_t& n, size_t& rows);

// --------------- SYCL NVIDIA ------------------------

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

// ----------------- Jacobi preconditioner --------------------

void compute_jacobi_preconditioner_buf(sycl::buffer<size_t>& row_ptr_buf, 
                                        sycl::buffer<size_t>& col_ind_buf, 
                                        sycl::buffer<double>& val_buf, 
                                        sycl::buffer<double>& M_inv_buf,
                                        size_t n, sycl::queue& q);

// -------------- Solvers -----------------

void CG_SYCL_jacobi(std::vector<double>& K_val, 
                std::vector<size_t>& K_col_ind, 
                std::vector<size_t>& K_row_ptr, 
                std::vector<double>& x, 
                std::vector<double>& b, 
                std::string& device);

// -------------------- Math operations --------------------

double scalar_product(const std::vector<double>& a, 
                    const std::vector<double>& b);

void scalar_product_parallel(sycl::queue& q, 
                            sycl::buffer<double>& a_buf, 
                            sycl::buffer<double>& b_buf, 
                            sycl::buffer<double>& res_buf,
                            size_t& n);

void CSR_mat_vec_prod_parallel(sycl::queue& q,
                            sycl::buffer<double>& vec_buf,
                            sycl::buffer<size_t>& row_ptr_buf,
                            sycl::buffer<size_t>& col_ind_buf,
                            sycl::buffer<double>& val_buf,
                            sycl::buffer<double>& res_buf,
                            size_t& n);