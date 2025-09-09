use std::fs::File;
use std::path::Path;
use sprs::io::read_matrix_market;
use sprs::{TriMatI};
use crate::graph::Graph;

/// Read a matrix market file and output Graph struct.
pub fn read_matrix_market_as_graph(file_path: &Path) -> Graph{
    // Attempt to read the matrix market file as a TriMat with edge lengths as f64.
    let tri_matrix_f64: Result<TriMatI<f64, usize>, _> = read_matrix_market(file_path);

    match tri_matrix_f64 {

        Ok(tri_matrix) => {
            // Read was successful, we return it after converting to CSR.
            let csr_matrix = tri_matrix.to_csr();
            Graph {
                graph_csr: csr_matrix,
            }
        },
        Err(E) => {
            // Read was unsuccessful, hence we try reading as a TriMat with edge lengths as i64.
            let tri_matrix_i64: TriMatI<i64, usize> = read_matrix_market(file_path)
                .expect("Failed to read matrix market file as both f64 and i64.");
            let rows = tri_matrix_i64.rows();
            let cols = tri_matrix_i64.cols();
            let mut tri_matrix_f64 = TriMatI::new((rows, cols));
            let iters = tri_matrix_i64.triplet_iter();
            let row_indices = iters.clone().into_row_inds();
            let col_indices = iters.clone().into_col_inds();
            let data = iters.clone().into_data();

            // Convert the edge lengths of TriMat from i64 to f64.
            for ((row, col), value) in row_indices.zip(col_indices).zip(data) {
                tri_matrix_f64.add_triplet(*row, *col, *value as f64);
            }
            // Convert to CSR and return it.
            Graph {
                graph_csr: tri_matrix_f64.to_csr(),
            }
        }
    }
}

/// Write the partition array to a file.
pub fn write_partition_data_to_file(partition: &[usize], file_name: &str) -> Result<(), std::io::Error> {
    let mut file = File::create(file_name)?;
    for vertex_id in 0..partition.len() {
        writeln!(file, "vertex {} => partition {}", vertex_id, partition[vertex_id])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;
    use tempfile::tempdir;
    use crate::input::read_matrix_market_as_graph;

    fn create_mock_file(dir: &Path, filename: &str, content: &str) -> String {
        let file_path = dir.join(filename);
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file_path.to_str().unwrap().to_string()
    }

    #[test]
    fn test_read_matrix_market_for_integer() -> Result<(), std::io::Error> {
        let temp_dir = tempdir()?;

        let f64_content = "\
                                %%MatrixMarket matrix coordinate integer general
                                %
                                5 5 3
                                1 1 1
                                2 2 2
                                5 5 5";
        let f64_file_path = create_mock_file(temp_dir.path(), "f64_matrix.mtx", f64_content);

        let graph_f64 = read_matrix_market_as_graph(&Path::new(&f64_file_path));

        // Assert that the graph was created correctly
        assert_eq!(graph_f64.graph_csr.rows(), 5);
        assert_eq!(graph_f64.graph_csr.cols(), 5);
        assert_eq!(graph_f64.graph_csr.nnz(), 3);

        Ok(())
    }

    #[test]
    fn test_read_matrix_market_for_real() -> Result<(), std::io::Error> {
        let temp_dir = tempdir()?;

        let f64_content = "\
                                %%MatrixMarket matrix coordinate real general
                                %
                                5 5 3
                                1 1 1.0
                                2 2 2.0
                                5 5 5.0";
        let f64_file_path = create_mock_file(temp_dir.path(), "f64_matrix.mtx", f64_content);
        let graph_f64 = read_matrix_market_as_graph(Path::new(&f64_file_path));

        // Assert that the graph was created correctly
        assert_eq!(graph_f64.graph_csr.rows(), 5);
        assert_eq!(graph_f64.graph_csr.cols(), 5);
        assert_eq!(graph_f64.graph_csr.nnz(), 3);

        Ok(())
    }
}