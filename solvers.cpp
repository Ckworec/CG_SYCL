#include "include.hpp"

void CG_SYCL_jacobi(std::vector<double>& K_val, 
                std::vector<size_t>& K_col_ind, 
                std::vector<size_t>& K_row_ptr, 
                std::vector<double>& x, 
                std::vector<double>& b, 
                std::string& device)
{
    sycl::queue cpu_queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
    sycl::queue gpu_queue(nvidia_selector, sycl::property::queue::in_order());
    sycl::queue q = (device == "CPU") ? cpu_queue : gpu_queue;

    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    double b_norm = scalar_product(b, b);

    if (b_norm < 1e-10)
    {
        std::fill(x.begin(), x.end(), 0.0);
        std::cout << "||b|| = 0" << std::endl;
        return;
    }

    size_t n = K_row_ptr.size() - 1;

    std::vector<double> r = b;
    std::vector<double> Ap(n);
    std::vector<double> M_inv;
    std::vector<double> z(n);
    std::vector<double> p = z;

    unsigned int iteration = 0;
    unsigned int max_iter = K_row_ptr.size();

    double old_rr = 0.0;
    double new_rr = 0.0;
    double pAp = 0.0;
    double alpha = 0.0;
    double beta = 0.0;

    sycl::buffer<double, 1> val_buf(K_val.data(), sycl::range<1>(K_val.size()));
    sycl::buffer<size_t, 1> col_ind_buf(K_col_ind.data(), sycl::range<1>(K_col_ind.size()));
    sycl::buffer<size_t, 1> row_ptr_buf(K_row_ptr.data(), sycl::range<1>(K_row_ptr.size()));
    sycl::buffer<double, 1> r_buf(r.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> z_buf(z.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> M_inv_buf(M_inv.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> p_buf(p.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> x_buf(x.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> Ap_buf(Ap.data(), sycl::range<1>(n));

    sycl::buffer<double, 1> old_rr_buf(&old_rr, sycl::range<1>(1));
    sycl::buffer<double, 1> new_rr_buf(&new_rr, sycl::range<1>(1));
    sycl::buffer<double, 1> pAp_buf(&pAp, sycl::range<1>(1));
    sycl::buffer<double, 1> alpha_buf(&alpha, sycl::range<1>(1));
    sycl::buffer<double, 1> beta_buf(&beta, sycl::range<1>(1));

    compute_jacobi_preconditioner_buf(row_ptr_buf, col_ind_buf, val_buf, M_inv_buf, n, q);

    q.submit([&](sycl::handler &h){
        sycl::accessor z_access(z_buf, h, sycl::read_write);
        sycl::accessor M_inv_access(M_inv_buf, h, sycl::read_only);
        sycl::accessor r_access(r_buf, h, sycl::read_only);
        sycl::accessor p_access(p_buf, h, sycl::write_only, sycl::no_init);

        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            z_access[i] = M_inv_access[i] * r_access[i];
            p_access[i] = z_access[i];
        });
    }).wait();

    scalar_product_parallel(q, r_buf, z_buf, old_rr_buf, n);

    do {
        iteration++;

        CSR_mat_vec_prod_parallel(q, p_buf, row_ptr_buf, col_ind_buf, val_buf, Ap_buf, n);

        scalar_product_parallel(q, p_buf, Ap_buf, pAp_buf, n);

        q.submit([&](sycl::handler& h) {
            sycl::accessor old_rr_access(old_rr_buf, h, sycl::read_only);
            sycl::accessor pAp_access(pAp_buf, h, sycl::read_write);
            sycl::accessor alpha_access(alpha_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor new_rr_access(new_rr_buf, h, sycl::write_only, sycl::no_init);

            h.single_task([=]() {
                alpha_access[0] = old_rr_access[0] / pAp_access[0];
                new_rr_access[0] = 0.0;
                pAp_access[0] = 0.0;
            });
        }).wait();

        q.submit([&](sycl::handler& h) {
            sycl::accessor x_access(x_buf, h, sycl::read_write);
            sycl::accessor r_access(r_buf, h, sycl::read_write);
            sycl::accessor z_access(z_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor p_access(p_buf, h, sycl::read_only);
            sycl::accessor Ap_access(Ap_buf, h, sycl::read_only);
            sycl::accessor alpha_access(alpha_buf, h, sycl::read_only);
            sycl::accessor M_inv_access(M_inv_buf, h, sycl::read_only);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                x_access[i] += alpha_access[0] * p_access[i];
                r_access[i] -= alpha_access[0] * Ap_access[i];
                z_access[i] = M_inv_access[i] * r_access[i];
            });
        }).wait();

        scalar_product_parallel(q, r_buf, z_buf, new_rr_buf, n);

        q.submit([&](sycl::handler& h) {
            sycl::accessor acc(new_rr_buf, h, sycl::read_only);
            h.copy(acc, &new_rr);
        }).wait_and_throw();

        if (new_rr / b_norm < eps * eps || iteration == max_iter)
            break;

        q.submit([&](sycl::handler& h) {
            sycl::accessor old_rr_access(old_rr_buf, h, sycl::read_write);
            sycl::accessor new_rr_access(new_rr_buf, h, sycl::read_only);
            sycl::accessor beta_access(beta_buf, h, sycl::write_only, sycl::no_init);

            h.single_task([=]() {
                beta_access[0] = new_rr_access[0] / old_rr_access[0];
                old_rr_access[0] = new_rr_access[0];
            });
        }).wait();

        q.submit([&](sycl::handler& h) {
            sycl::accessor p_access(p_buf, h, sycl::read_write);
            sycl::accessor z_access(z_buf, h, sycl::read_only);
            sycl::accessor beta_access(beta_buf, h, sycl::read_only);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                p_access[i] = z_access[i] + beta_access[0] * p_access[i];
            });
        }).wait();
    } while (new_rr / b_norm > eps * eps && iteration < max_iter);

    q.submit([&](sycl::handler& h) {
        sycl::accessor acc(x_buf, h, sycl::read_only);
        h.copy(acc, x.data());
    }).wait_and_throw();

    std::cout << "iterations: " << iteration << std::endl;
}
