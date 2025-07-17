using namespace std;

template<typename T>
CSR_matrix<T>::CSR_matrix() {
    m_row_ptr.push_back(0);
    m_row_ptr.push_back(1);
    m_col_ind.push_back(0);
    m_val.push_back(1);
    m_rows = 1;
    m_cols = 1;
    m_nnz = 1;
}

template<typename T>
CSR_matrix<T>::CSR_matrix(vector<size_t>& row_ptr, vector<size_t>& col_ind, vector<T>& val) {
    m_row_ptr = row_ptr;
    m_col_ind = col_ind;
    m_val = val;
    m_rows = row_ptr.size();
    m_cols = row_ptr.size();
    m_nnz = val.size();
}

template<typename T>
void CSR_matrix<T>::read_mat(const string& name_file) {
    std::ifstream mat_file(name_file);

    mat_file >> m_rows >> m_nnz;
    m_cols = m_rows;

    m_row_ptr.resize(m_rows);
    m_col_ind.resize(m_nnz);
    m_val.resize(m_nnz);

    for (size_t i = 0; i <= m_rows; ++i) {
        mat_file >> m_row_ptr[i];
    }

    for (size_t i = 0; i < m_nnz; ++i) {
        mat_file >> m_col_ind[i];
    }
    
    for (size_t i = 0; i < m_nnz; ++i) {
        mat_file >> m_val[i];
    }

    mat_file.close();
}

template<typename T>
CSR_matrix<T>::~CSR_matrix() {
    m_row_ptr.clear();
    m_col_ind.clear();
    m_val.clear();
}

template<typename T>
vector<size_t> CSR_matrix<T>::take_row_ptr() {
    return m_row_ptr;
}

template<typename T>
vector<size_t> CSR_matrix<T>::take_col_ind() {
    return m_col_ind;
}

template<typename T>
vector<T> CSR_matrix<T>::take_val() {
    return m_val;
}

template<typename T>
size_t CSR_matrix<T>::take_rows() {
    return m_rows;
}

template<typename T>
size_t CSR_matrix<T>::take_nnz() {
    return m_nnz;
}