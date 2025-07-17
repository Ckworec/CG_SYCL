#include "func.hpp"

void CG_SYCL_jacobi(CSR_matrix<double>& mat, 
                std::vector<double>& x, 
                std::vector<double>& b, 
                std::string& device)
{
    double b_norm = scalar_product(b, b);

    if (b_norm < 1e-10)
    {
        std::fill(x.begin(), x.end(), 0.0);
        std::cout << "||b|| = 0" << std::endl;
        return;
    }

    sycl::queue cpu_queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
    sycl::queue gpu_queue(nvidia_selector, sycl::property::queue::in_order());
    sycl::queue q = (device == "CPU") ? cpu_queue : gpu_queue;

    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    size_t n = mat.take_row_ptr().size() - 1;

    std::vector<double> r = b;
    std::vector<double> Ap(n);
    std::vector<double> M_inv;
    std::vector<double> z(n);
    std::vector<double> p = z;
    std::vector<size_t> row_ptr = mat.take_row_ptr();
    std::vector<size_t> col_ind = mat.take_col_ind();
    std::vector<double> val = mat.take_val();

    unsigned int iteration = 0;
    unsigned int max_iter = mat.take_row_ptr().size();

    double old_rr = 0.0;
    double new_rr = 0.0;
    double pAp = 0.0;
    double alpha = 0.0;
    double beta = 0.0;

    sycl::buffer<double, 1> val_buf(val.data(), sycl::range<1>(val.size()));
    sycl::buffer<size_t, 1> col_ind_buf(col_ind.data(), sycl::range<1>(col_ind.size()));
    sycl::buffer<size_t, 1> row_ptr_buf(row_ptr.data(), sycl::range<1>(row_ptr.size()));
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

void GMRES_SYCL(CSR_matrix<double>& mat,
                std::vector<double>& x, 
                std::vector<double>& b, 
                std::string& device) 
{
    sycl::queue cpu_queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
    sycl::queue gpu_queue(nvidia_selector, sycl::property::queue::in_order());
    sycl::queue q = (device == "CPU") ? cpu_queue : gpu_queue;

    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    size_t n = mat.take_row_ptr().size() - 1;
    size_t max_iter = n;
    size_t iter = 0;

    std::vector<double> r0(n);
    std::vector<double> tmp(n);
    std::vector<size_t> row_ptr = mat.take_row_ptr();
    std::vector<size_t> col_ind = mat.take_col_ind();
    std::vector<double> val = mat.take_val();

    // r0 * r0
    double beta = 0.0;

    double resid = 0.0;
    double h_ij = 0.0;
    double h_next = 0.0;
    double h1 = 0.0;
    double h2 = 0.0;
    double r_val = 0.0;

    sycl::buffer<double, 1> val_buf(val.data(), sycl::range<1>(val.size()));
    sycl::buffer<size_t, 1> col_ind_buf(col_ind.data(), sycl::range<1>(col_ind.size()));
    sycl::buffer<size_t, 1> row_ptr_buf(row_ptr.data(), sycl::range<1>(row_ptr.size()));
    sycl::buffer<double, 1> r0_buf(r0.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> b_buf(b.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> x_buf(x.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> tmp_buf(tmp.data(), sycl::range<1>(n));

    sycl::buffer<double, 1> beta_buf(&beta, sycl::range<1>(1));
    sycl::buffer<double, 1> h_ij_buf(&h_ij, sycl::range<1>(1));
    sycl::buffer<double, 1> h_next_buf(&h_next, sycl::range<1>(1));
    sycl::buffer<double, 1> h1_buf(&h1, sycl::range<1>(1));
    sycl::buffer<double, 1> h2_buf(&h2, sycl::range<1>(1));
    sycl::buffer<double, 1> r_val_buf(&r_val, sycl::range<1>(1));
    sycl::buffer<double, 1> resid_buf(&resid, sycl::range<1>(1));

    CSR_mat_vec_prod_parallel(q, x_buf, row_ptr_buf, col_ind_buf, val_buf, tmp_buf, n);

    subtract(q, b_buf, tmp_buf, r0_buf, n);
    scalar_product_parallel(q, r0_buf, r0_buf, beta_buf, n);

    copy_from_nvidia(q, beta_buf, beta);

    if (beta < eps * eps) return;

    std::vector<double> V((max_iter + 1) * n, 0.0); // Крылов
    std::vector<double> H((max_iter + 1) * max_iter, 0.0); // Гессенберг
    std::vector<double> cs(max_iter);
    std::vector<double> sn(max_iter);
    std::vector<double> e1(max_iter + 1, 0.0);
    std::vector<double> w(n);
    e1[0] = beta;

    sycl::buffer<double, 1> V_buf(V.data(), sycl::range<1>((max_iter + 1) * n));
    sycl::buffer<double, 1> H_buf(H.data(), sycl::range<1>((max_iter + 1) * max_iter));
    sycl::buffer<double, 1> cs_buf(cs.data(), sycl::range<1>(max_iter));
    sycl::buffer<double, 1> sn_buf(sn.data(), sycl::range<1>(max_iter));
    sycl::buffer<double, 1> e1_buf(e1.data(), sycl::range<1>(max_iter + 1));
    sycl::buffer<double, 1> w_buf(w.data(), sycl::range<1>(n));

    // v0 = r / ||r||
    scale_division(q, r0_buf, beta_buf, V_buf, n);

    do {
        // w = A * V_j
        q.submit([&](sycl::handler& h){
            sycl::accessor V_access(V_buf, h, sycl::read_only);
            sycl::accessor row_ptr_access(row_ptr_buf, h, sycl::read_only);
            sycl::accessor col_ind_access(col_ind_buf, h, sycl::read_only);
            sycl::accessor val_access(val_buf, h, sycl::read_only);
            sycl::accessor w_access(w_buf, h, sycl::write_only, sycl::no_init);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
                double val(0.0);
                for (size_t k = row_ptr_access[i]; k < row_ptr_access[i + 1]; ++k) {
                    val += val_access[k] * V_access[iter * n + col_ind_access[k]];
                }
                w_access[i] = val;
            });
        }).wait();

        for (size_t k = 0; k <= iter; ++k) {
            q.submit([&](sycl::handler& h) {
                sycl::accessor V_access(V_buf, h, sycl::read_only);
                sycl::accessor w_access(w_buf, h, sycl::read_only);
                auto red_sum = sycl::reduction(h_ij_buf, h, sycl::plus<double>());

                h.parallel_for(sycl::range<1>(n), red_sum, [=](sycl::id<1> i, auto &sum) {
                    sum += V_access[k * n + i] * w_access[i];
                });
            }).wait();

            q.submit([&](sycl::handler& h) {
                sycl::accessor H_access(H_buf, h, sycl::read_write);
                sycl::accessor h_ij_access(h_ij_buf, h, sycl::read_only);

                h.single_task([=]() {
                    H_access[k * max_iter + iter] = h_ij_access[0];
                });
            }).wait();

            q.submit([&](sycl::handler& h){
                sycl::accessor w_access(w_buf, h, sycl::read_write);
                sycl::accessor h_ij_access(h_ij_buf, h, sycl::read_only);
                sycl::accessor V_access(V_buf, h, sycl::read_only);

                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
                    w_access[i] -= h_ij_access[0] * V_access[k * n + i];
                });
            }).wait();
        }

        scalar_product_parallel(q, w_buf, w_buf, h_next_buf, n);

        q.submit([&](sycl::handler& h) {
            sycl::accessor H_access(H_buf, h, sycl::read_write);
            sycl::accessor h_next_access(h_next_buf, h, sycl::read_only);

            h.single_task([=]() {
                H_access[(iter + 1) * max_iter + iter] = h_next_access[0];
            });
        }).wait();

        copy_from_nvidia(q, h_next_buf, h_next);

        if (h_next != 0.0) {
            q.submit([&](sycl::handler& h){
                sycl::accessor V_access(V_buf, h, sycl::write_only, sycl::no_init);
                sycl::accessor w_access(w_buf, h, sycl::read_only);
                sycl::accessor h_next_access(h_next_buf, h, sycl::read_only);

                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
                    V_access[(iter + 1) * n + i] = w_access[i] / h_next_access[0];
                });
            }).wait();
        }

        q.submit([&](sycl::handler& h){
            sycl::accessor H_access(H_buf, h, sycl::read_write);
            sycl::accessor sn_access(sn_buf, h, sycl::read_only);
            sycl::accessor cs_access(cs_buf, h, sycl::read_only);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
                double temp = cs_access[i] * H_access[i * max_iter + iter] + sn_access[i] * H_access[(i + 1) * max_iter + iter];
                H_access[(i + 1) * max_iter + iter] = - sn_access[i] * H_access[i * max_iter + iter] + cs_access[i] * H_access[(i + 1) * max_iter + iter];
                H_access[i * max_iter + iter] = temp;
            });
        }).wait();

        q.submit([&](sycl::handler& h) {
            sycl::accessor H_access(H_buf, h, sycl::read_write);
            sycl::accessor h1_access(h1_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor h2_access(h2_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor r_val_access(r_val_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor cs_access(cs_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor sn_access(sn_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor resid_access(resid_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor e1_access(e1_buf, h, sycl::read_write);

            h.single_task([=]() {
                h1_access[0] = H_access[iter * max_iter + iter];
                h2_access[0] = H_access[(iter + 1) * max_iter + iter];
                r_val_access[0] = sycl::sqrt(h1_access[0] * h1_access[0] + h2_access[0] * h2_access[0]);
                sn_access[iter] = h2_access[0] / r_val_access[0];
                cs_access[iter] = h1_access[0] / r_val_access[0];
                H_access[iter * max_iter + iter] = r_val_access[0];
                H_access[(iter + 1) * max_iter + iter] = r_val_access[0];
                double temp = cs_access[iter] * e1_access[iter] + sn_access[iter] * e1_access[iter + 1];
                e1_access[iter + 1] = - sn_access[iter] * e1_access[iter] + cs_access[iter] * e1_access[iter + 1];
                e1_access[iter] = temp;
                resid_access[0] = sycl::fabs(e1_access[iter + 1]);
            });
        }).wait();

        copy_from_nvidia(q, resid_buf, resid);

        std::cout << "Iter " << iter + 1 << ", residual = " << resid << std::endl;

        if (resid < eps) {
            std::vector<double> y(iter + 1);

            copy_from_nvidia(q, H_buf, H);

            for (size_t i = iter; i >= 0; --i) {
                y[i] = e1[i];
                for (size_t k = i + 1; k <= iter; ++k)
                    y[i] -= H[i * max_iter + k] * y[k];
                y[i] /= H[i * max_iter + i];
            }

            sycl::buffer<double, 1> y_buf(y.data(), sycl::range<1>(iter + 1));

            q.submit([&](sycl::handler& h) {
                sycl::accessor x_access(x_buf, h, sycl::write_only, sycl::no_init);
                sycl::accessor y_access(y_buf, h, sycl::read_only);
                sycl::accessor V_access(V_buf, h, sycl::read_only);

                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    double sum = 0.0;
                    for (size_t k = 0; k <= iter; ++k) {
                        sum += V_access[k * n + i] * y_access[k];
                    }
                    x_access[i] += sum;
                });
            }).wait();

            copy_from_nvidia(q, x_buf, x);

            return;
        }

        iter++;

    } while (resid > eps && iter < max_iter);

    std::vector<double> y(max_iter);

    for (size_t i = max_iter - 1; i >= 0; --i) {
        y[i] = e1[i];
        for (size_t k = i + 1; k < max_iter; ++k)
            y[i] -= H[i * max_iter + k] * y[k];
        y[i] /= H[i * max_iter + i];
    }

    sycl::buffer<double, 1> y_buf(y.data(), sycl::range<1>(max_iter));

    q.submit([&](sycl::handler& h) {
        sycl::accessor x_access(x_buf, h, sycl::write_only, sycl::no_init);
        sycl::accessor y_access(y_buf, h, sycl::read_only);
        sycl::accessor V_access(V_buf, h, sycl::read_only);

        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            double sum = 0.0;
            for (size_t k = 0; k < max_iter; ++k) {
                sum += V_access[k * n + i] * y_access[k];
            }
            x_access[i] += sum;
        });
    }).wait();

    copy_from_nvidia(q, x_buf, x);

    std::cout << "iterations: " << iter << std::endl;
}

void CGS_SYCL(CSR_matrix<double>& mat,
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

    size_t n = mat.take_row_ptr().size() - 1;

    std::vector<double> r = b;
    std::vector<double> Ap(n);
    std::vector<double> p = r;
    std::vector<double> u = r;
    std::vector<double> r_start = r;
    std::vector<double> q_per(n);
    std::vector<double> v(n);
    std::vector<size_t> row_ptr = mat.take_row_ptr();
    std::vector<size_t> col_ind = mat.take_col_ind();
    std::vector<double> val = mat.take_val();

    unsigned int iteration = 0;
    unsigned int max_iter = mat.take_row_ptr().size();

    double old_rr = 0.0;
    double new_rr = 0.0;
    double alpha = 0.0;
    double beta = 0.0;

    sycl::buffer<double, 1> val_buf(val.data(), sycl::range<1>(val.size()));
    sycl::buffer<size_t, 1> col_ind_buf(col_ind.data(), sycl::range<1>(col_ind.size()));
    sycl::buffer<size_t, 1> row_ptr_buf(row_ptr.data(), sycl::range<1>(row_ptr.size()));
    sycl::buffer<double, 1> r_buf(r.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> r_start_buf(r_start.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> p_buf(p.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> x_buf(x.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> u_buf(u.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> Ap_buf(Ap.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> q_buf(q_per.data(), sycl::range<1>(n));
    sycl::buffer<double, 1> v_buf(v.data(), sycl::range<1>(n));

    sycl::buffer<double, 1> old_rr_buf(&old_rr, sycl::range<1>(1));
    sycl::buffer<double, 1> new_rr_buf(&new_rr, sycl::range<1>(1));
    sycl::buffer<double, 1> alpha_buf(&alpha, sycl::range<1>(1));
    sycl::buffer<double, 1> beta_buf(&beta, sycl::range<1>(1));

    do {
        scalar_product_parallel(q, r_buf, r_start_buf, old_rr_buf, n);

        if (iteration == 0) {
            q.submit([&](sycl::handler& h) {
                sycl::accessor u_access(u_buf, h, sycl::write_only, sycl::no_init);
                sycl::accessor r_access(r_buf, h, sycl::read_only);

                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    u_access[i] = r_access[i];
                });
            }).wait();
        } else {
            scalar_product_parallel(q, r_buf, r_start_buf, beta_buf, n);

            q.submit([&](sycl::handler& h){
                sycl::accessor p_access(p_buf, h, sycl::read_write);
                sycl::accessor u_access(u_buf, h, sycl::write_only, sycl::no_init);
                sycl::accessor r_access(r_buf, h, sycl::read_only);
                sycl::accessor q_access(q_buf, h, sycl::read_only);
                sycl::accessor beta_access(beta_buf, h, sycl::read_only);

                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
                    u_access[i] = r_access[i] + beta_access[0] * q_access[i];
                    p_access[i] = u_access[i] + beta_access[0] * (q_access[i] + beta_access[0] * p_access[i]);
                });
            }).wait();
        }

        CSR_mat_vec_prod_parallel(q, p_buf, row_ptr_buf, col_ind_buf, val_buf, Ap_buf, n);

        scalar_product_parallel(q, r_start_buf, Ap_buf, new_rr_buf, n);

        iteration++;

        q.submit([&](sycl::handler& h) {
            sycl::accessor old_rr_access(old_rr_buf, h, sycl::read_write);
            sycl::accessor alpha_access(alpha_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor new_rr_access(new_rr_buf, h, sycl::read_write);

            h.single_task([=]() {
                alpha_access[0] = old_rr_access[0] / new_rr_access[0];
                new_rr_access[0] = 0.0;
                old_rr_access[0] = 0.0;
            });
        }).wait();

        q.submit([&](sycl::handler& h) {
            sycl::accessor x_access(x_buf, h, sycl::read_write);
            sycl::accessor q_access(q_buf, h, sycl::read_write);
            sycl::accessor u_access(u_buf, h, sycl::read_only);
            sycl::accessor Ap_access(Ap_buf, h, sycl::read_only);
            sycl::accessor alpha_access(alpha_buf, h, sycl::read_only);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                q_access[i] = u_access[i] - alpha_access[0] * Ap_access[i];
                x_access[i] += alpha_access[0] * (u_access[i] + q_access[i]);
            });
        }).wait();

        q.submit([&](sycl::handler& h) {
            sycl::accessor q_access(q_buf, h, sycl::read_write);
            sycl::accessor u_access(u_buf, h, sycl::read_only);
            sycl::accessor alpha_access(alpha_buf, h, sycl::read_only);
            sycl::accessor v_access(v_buf, h, sycl::write_only, sycl::no_init);
            sycl::accessor row_ptr_access(row_ptr_buf, h, sycl::read_only);
            sycl::accessor col_ind_access(col_ind_buf, h, sycl::read_only);
            sycl::accessor val_access(val_buf, h, sycl::read_only);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                double val = 0.0;
                for (size_t k = row_ptr_access[i]; k < row_ptr_access[i + 1]; ++k) {
                    val += val_access[k] * (u_access[col_ind_access[k]] + q_access[col_ind_access[k]]);
                }
                v_access[i] = val;
            });
        }).wait();

        q.submit([&](sycl::handler& h) {
            sycl::accessor r_access(r_buf, h, sycl::read_write);
            sycl::accessor v_access(v_buf, h, sycl::read_only);
            sycl::accessor alpha_access(alpha_buf, h, sycl::read_only);

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                r_access[i] -= alpha_access[0] * v_access[i];
            });
        }).wait();

        scalar_product_parallel(q, r_buf, r_buf, new_rr_buf, n);

        copy_from_nvidia(q, new_rr_buf, new_rr);
    } while (new_rr / b_norm > eps * eps && iteration < max_iter);

    copy_from_nvidia(q, x_buf, x);

    std::cout << "iterations: " << iteration << std::endl;
}