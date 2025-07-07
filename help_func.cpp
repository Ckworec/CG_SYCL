#include "include.hpp"

void read(std::vector<size_t>& row_ptr, 
        std::vector<size_t>& col_ind, 
        std::vector<double>& val, 
        std::vector<double>& vec,
        std::ifstream& vec_file,
        std::ifstream& mat_file, 
        size_t& nnz, size_t& rows)
{
    for (size_t i = 0; i <= rows; ++i){
        mat_file >> row_ptr[i];
    }

    for (size_t i = 0; i < nnz; ++i){
        mat_file >> col_ind[i];
        vec_file >> vec[i];
    }
    
    for (size_t i = 0; i < nnz; ++i){
        mat_file >> val[i];
    }
}