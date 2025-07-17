#include "class.hpp"

#define eps 1.e-4

// --------------- Help functions ---------------------

void read_vec(std::vector<double>& vec,
            std::string& name_file,
            size_t nnz);

void test_CG_SYCL_jacobi(CSR_matrix<double>& mat, 
                        std::vector<double>& x, 
                        std::vector<double>& b, 
                        std::string& device);

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

void copy_from_nvidia(sycl::queue& q, 
                        sycl::buffer<double>& a_buf, 
                        double& a);

void copy_from_nvidia(sycl::queue& q, 
                        sycl::buffer<double>& a_buf, 
                        std::vector<double>& a);

// ----------------- Preconditioners --------------------

void compute_jacobi_preconditioner_buf(sycl::buffer<size_t>& row_ptr_buf, 
                                        sycl::buffer<size_t>& col_ind_buf, 
                                        sycl::buffer<double>& val_buf, 
                                        sycl::buffer<double>& M_inv_buf,
                                        size_t n, sycl::queue& q);

// -------------- Solvers -----------------

void CG_SYCL_jacobi(CSR_matrix<double>& mat,
                std::vector<double>& x, 
                std::vector<double>& b, 
                std::string& device);

void GMRES_SYCL(CSR_matrix<double>& mat,
            std::vector<double>& x, 
            std::vector<double>& b, 
            std::string& device); // не проверен

void CGS_SYCL(CSR_matrix<double>& mat,
            std::vector<double>& x, 
            std::vector<double>& b, 
            std::string& device); // не проверен

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

// res = alpha * vec
void scale(sycl::queue&q,
            sycl::buffer<double>& vec_buf,
            sycl::buffer<double>& alpha_buf,
            sycl::buffer<double>& res_buf,
            size_t& n);

// res = alpha * x + y
void axpy(sycl::queue& q,
            sycl::buffer<double>& x_buf,
            sycl::buffer<double>& y_buf,
            sycl::buffer<double>& alpha_buf,
            sycl::buffer<double>& res_buf,
            size_t& n);

// res = x - y
void subtract(sycl::queue& q,
            sycl::buffer<double>& x_buf,
            sycl::buffer<double>& y_buf,
            sycl::buffer<double>& res_buf,
            size_t& n);

void scale_division(sycl::queue& q,
                    sycl::buffer<double>& x_buf,
                    sycl::buffer<double>& alpha_buf,
                    sycl::buffer<double>& res_buf,
                    size_t& n);