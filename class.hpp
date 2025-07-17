#include <stdio.h>
#include <math.h>
#include <thread>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sycl/sycl.hpp>

using namespace std;

template<typename T>
class CSR_matrix {
    private:
        vector<size_t> m_row_ptr;
        vector<size_t> m_col_ind;
        vector<T> m_val;
        int m_rows, m_cols, m_nnz; 

    public:
        CSR_matrix(vector<size_t>& row_ptr, vector<size_t>& col_ind, vector<T>& val);
        CSR_matrix();

        ~CSR_matrix();

        void read_mat(const string& name_file);
        vector<size_t> take_row_ptr();
        vector<size_t> take_col_ind();
        vector<T> take_val();
        size_t take_rows();
        size_t take_nnz();
};

#include "class.tpp"