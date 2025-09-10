# GraphCut Graph Partitioner

 GraphCut is a multilevel-multithreaded 2-way graph partitioner built in Rust. It takes in as input a graph (denoted by a sparse matrix) in matrix market format and partitions it into 2 partitions with the goal of minimizing edge cut is and balancing the weights of the vertices across the two partitions.
 It uses the following 3 algorithms for generating the partitions.

- **Heavy Edge Matching** algorithm for graph coarsening.
- **Recursive Co-ordinate Bisection** for initial partition.
- **Jet Partition Refiner** for refining the partition during the uncoarsening phase.

## Usage
```rust
use std::path::Path;
use GraphCut::io::{read_matrix_market_as_graph, write_partition_data_to_file};
use GraphCut::gen_weights::gen_random_weights;
use GraphCut::algorithms::MultiLevelPartitioner;
use GraphCut::imbalance::imbalance;
use GraphCut::Partition;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = read_matrix_market_as_graph(Path::new("vt2010.mtx"));
    let weights = gen_random_weights(graph.len(), 1.0, 3.0);
    let mut partition = vec![0; graph.len()];
    
    MultiLevelPartitioner {..Default::default()}.partition(&mut partition, (graph.clone(), &weights))?;

    let edge_cut = graph.edge_cut(&partition);

    println!("Edge cut = {:?}", edge_cut);
    println!("Weights = {:?}", imbalance(2, &partition, weights.clone()));
    write_partition_data_to_file(&partition, "vt2010_partition")?;
    
    Ok(())
}
```
## References
- Berger, and Bokhari. "A partitioning strategy for nonuniform problems on multiprocessors." IEEE Transactions on Computers 100, no. 5 (1987): 570-580.
- Bramas, Berenger. "A novel hybrid quicksort algorithm vectorized using AVX-512 on Intel Skylake." arXiv preprint arXiv:1704.08579 (2017).
- Gilbert, Michael S., Kamesh Madduri, Erik G. Boman, and Siva Rajamanickam. "Jet: Multilevel graph partitioning on graphics processing units." SIAM Journal on Scientific Computing 46, no. 5 (2024): B700-B724.

## Credits

This project includes code from another graph partitioning project called Coupe.

* **Original Project:** https://github.com/LIHPC-Computational-Geometry/coupe
* **Author(s):** Hubert Hirtz, Cédric Chevalier, SébastienM
* **License:** MIT License
