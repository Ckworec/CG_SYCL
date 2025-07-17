#include "func.hpp"

double scalar_product(const std::vector<double>& a, 
                    const std::vector<double>& b)
{
    size_t n = a.size();
    double result = 0.0;

    for (int i = 0; i < n; ++i){
        result += a[i] * b[i];
    }
    
    return result;
}

void scalar_product_parallel(sycl::queue& q, 
                            sycl::buffer<double>& a_buf, 
                            sycl::buffer<double>& b_buf, 
                            sycl::buffer<double>& res_buf,
                            size_t& n)
{
    q.submit([&](sycl::handler& h){
        sycl::accessor a_access(a_buf, h, sycl::read_only);
        sycl::accessor b_access(b_buf, h, sycl::read_only);

        auto red_sum = sycl::reduction(res_buf, h, sycl::plus<double>());

        h.parallel_for(sycl::range<1>(n), red_sum, [=](sycl::id<1> i, auto &sum) {
                sum += a_access[i] * b_access[i];
            }
        );
    }).wait();
}

void CSR_mat_vec_prod_parallel(sycl::queue& q,
                            sycl::buffer<double>& vec_buf,
                            sycl::buffer<size_t>& row_ptr_buf,
                            sycl::buffer<size_t>& col_ind_buf,
                            sycl::buffer<double>& val_buf,
                            sycl::buffer<double>& res_buf,
                            size_t& n)
{
    q.submit([&](sycl::handler& h){
        sycl::accessor vec_access(vec_buf, h, sycl::read_only);
        sycl::accessor row_ptr_access(row_ptr_buf, h, sycl::read_only);
        sycl::accessor col_ind_access(col_ind_buf, h, sycl::read_only);
        sycl::accessor val_access(val_buf, h, sycl::read_only);
        sycl::accessor res_access(res_buf, h, sycl::write_only, sycl::no_init);

        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
            double val(0.0);
            for (size_t k = row_ptr_access[i]; k < row_ptr_access[i + 1]; ++k) {
                val += val_access[k] * vec_access[col_ind_access[k]];
            }
            res_access[i] = val;
        });
    }).wait();       
}

void scale(sycl::queue&q,
            sycl::buffer<double>& vec_buf,
            sycl::buffer<double>& alpha_buf,
            sycl::buffer<double>& res_buf,
            size_t& n)
{
    q.submit([&](sycl::handler& h){
        sycl::accessor vec_access(vec_buf, h, sycl::read_write);
        sycl::accessor alpha_access(alpha_buf, h, sycl::read_only);
        sycl::accessor res_access(res_buf, h, sycl::write_only, sycl::no_init);

        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
            res_access[i] = vec_access[i] * alpha_access[0];
        });
    }).wait();
}

void axpy(sycl::queue& q,
        sycl::buffer<double>& x_buf,
        sycl::buffer<double>& y_buf,
        sycl::buffer<double>& alpha_buf,
        sycl::buffer<double>& res_buf,
        size_t& n)
{
    q.submit([&](sycl::handler& h){
        sycl::accessor x_access(x_buf, h, sycl::read_only);
        sycl::accessor y_access(y_buf, h, sycl::read_write);
        sycl::accessor res_access(res_buf, h, sycl::write_only, sycl::no_init);
        sycl::accessor alpha_access(alpha_buf, h, sycl::read_only);

        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
            res_access[i] = alpha_access[0] * x_access[i] + y_access[i];
        });
    }).wait();
}

void subtract(sycl::queue& q,
            sycl::buffer<double>& x_buf,
            sycl::buffer<double>& y_buf,
            sycl::buffer<double>& res_buf,
            size_t& n)
{
    q.submit([&](sycl::handler& h){
        sycl::accessor x_access(x_buf, h, sycl::read_only);
        sycl::accessor y_access(y_buf, h, sycl::read_only);
        sycl::accessor res_access(res_buf, h, sycl::write_only, sycl::no_init);

        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
            res_access[i] = x_access[i] - y_access[i];
        });
    }).wait();
}

void scale_division(sycl::queue& q,
                    sycl::buffer<double>& x_buf,
                    sycl::buffer<double>& alpha_buf,
                    sycl::buffer<double>& res_buf,
                    size_t& n)
{
    q.submit([&](sycl::handler& h){
        sycl::accessor x_access(x_buf, h, sycl::read_only);
        sycl::accessor alpha_access(alpha_buf, h, sycl::read_only);
        sycl::accessor res_access(res_buf, h, sycl::write_only, sycl::no_init);

        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
            res_access[i] = x_access[i] / alpha_access[0];
        });
    }).wait();
}