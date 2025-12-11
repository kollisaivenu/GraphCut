# GraphCut Graph Partitioner

 GraphCut is a multilevel k-way graph partitioner built in Rust. It takes in as input a graph (denoted by a sparse matrix) in matrix market format and partitions it into 2 partitions with the goal of minimizing edge cut is and balancing the weights of the vertices across the two partitions.
 It uses the following 3 algorithms for generating the partitions.

- **Heavy Edge Matching** algorithm for graph coarsening.
- **Simple Greedy Bucketing Algorithm** for initial partition.
- **Jet Partition Refiner** for refining the partition during the uncoarsening phase.

## Usage

### Use Programmatically
```rust
use std::path::Path;
use GraphCut::io::{read_matrix_market_as_graph, write_partition_data_to_file};
use GraphCut::gen_weights::gen_random_weights;
use GraphCut::algorithms::MultiLevelPartitioner;
use GraphCut::imbalance::imbalance;
use GraphCut::Partition;

fn main() -> Result<(), Box<dyn std::error::Error>> { 
    let graph = read_matrix_market_as_graph(Path::new("./testdata/vt2010.mtx"))?;
    let weights = gen_random_weights(graph.len(), 1, 3);
    let mut partition = vec![0; graph.len()];
    let num_of_partitions = 2;
    MultiLevelPartitioner {num_of_partitions, ..Default::default()}.partition(&mut partition, (graph.clone(), &weights))?;
    let edge_cut = graph.edge_cut(&partition);
    println!("Edge cut = {:?}", edge_cut);
    println!("Imbalance ratio = {:?}", imbalance(num_of_partitions, &partition, &weights));
    write_partition_data_to_file(&partition, "vt2010_partition")?;

    Ok(())
}
```
### Use as CLI Tool
1. Clone the repository:
   ```bash
   git clone https://github.com/kollisaivenu/GraphCut.git
   ```
2. Navigate the directory:
   ```bash
   cd GraphCut
   ```
3. Install the tool
   ```bash
   cargo install --path .
   ```
4. Syntax is as follows:
   ```bash
   GraphCut <MTX_FILEPATH> <NUM_OF_PARTITIONS> <BALANCE_FACTOR> <PARTITION_FILE>[OPTIONS]
   ```
5. Example Usage
   ```bash
   GraphCut "./testdata/vt2010.mtx" 2 0.1 "vt2010_partition"
   ```

## References
- Horowitz, Ellis and Sahni, Sartaj, 1974. Computing partitions with applications to the knapsack problem. *J. ACM*, 21(2):277–292.
- Gilbert, Michael S., Kamesh Madduri, Erik G. Boman, and Siva Rajamanickam. "Jet: Multilevel graph partitioning on graphics processing units." SIAM Journal on Scientific Computing 46, no. 5 (2024): B700-B724.

## Credits

This project was the outcome of an Independent Study done under the supervision of **Professor Jed Brown at University of Colorado, Boulder**.

This project includes code from another graph partitioning project called Coupe (See [NOTICE.md](NOTICE.md) file).

* **Original Project:** https://github.com/LIHPC-Computational-Geometry/coupe
* **Author(s):** Hubert Hirtz, Cédric Chevalier, SébastienM
* **License:** MIT License
