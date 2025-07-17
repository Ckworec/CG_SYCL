#include "func.hpp"

void compute_jacobi_preconditioner_buf(sycl::buffer<size_t>& row_ptr_buf, 
                                        sycl::buffer<size_t>& col_ind_buf, 
                                        sycl::buffer<double>& val_buf, 
                                        sycl::buffer<double>& M_inv_buf,
                                        size_t n, sycl::queue& q)
{
    q.submit([&](sycl::handler& h) {
        sycl::accessor val_access(val_buf, h, sycl::read_only);
        sycl::accessor row_ptr_access(row_ptr_buf, h, sycl::read_only);
        sycl::accessor col_ind_access(col_ind_buf, h, sycl::read_only);
        sycl::accessor M_access(M_inv_buf, h, sycl::write_only, sycl::no_init);

        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            for (size_t j = row_ptr_access[i]; j < row_ptr_access[i + 1]; ++j)
            {
                if(col_ind_access[j] == i){
                    if (std::abs(val_access[j]) > 1e-14)
                        M_access[i] = 1.0 / val_access[j];
                    break;
                }
            }
        });
    }).wait();
}