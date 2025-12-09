use std::path::Path;
use std::time::Instant;
use GraphCut::algorithms::{MultiLevelPartitioner};
use GraphCut::gen_weights::gen_uniform_weights;
use GraphCut::imbalance::imbalance;
use GraphCut::io::read_matrix_market_as_graph;
use GraphCut::Partition;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = read_matrix_market_as_graph(Path::new("./testdata/vt2010.mtx"))?;
    let weights = gen_uniform_weights(graph.len());
    let mut partition = vec![0; graph.len()];
    let start = Instant::now();
    MultiLevelPartitioner {jet_iterations: 12, num_of_partitions: 2, jet_tolerance_factor: 0.999, ..Default::default()}.partition(&mut partition, (graph.clone(), &weights))?;
    let elapsed_time = start.elapsed();
    let edge_cut = graph.edge_cut(&partition);
    let imbalance_of_partition = imbalance(2, &partition, &weights);
    println!("Edge cut {:?}", edge_cut);
    println!("Imbalance {:?}", imbalance_of_partition);
    println!("Execution time {:?}", elapsed_time);
    Ok(())
}