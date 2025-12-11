use std::path::Path;
use std::time::Instant;
use GraphCut::algorithms::{MultiLevelPartitioner};
use GraphCut::gen_weights::gen_uniform_weights;
use GraphCut::imbalance::imbalance;
use GraphCut::io::{read_matrix_market_as_graph, write_partition_data_to_file};
use GraphCut::Partition;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path of the .mtx file
    mtx_filepath: String,

    /// Number of Partitions
    num_of_partitions: usize,

    /// Imbalance Ratio
    balance_factor: f64,

    /// Filename where the partition mapping can be stored
    partition_file: String,

    /// Number of Jet Partitioner Iterations
    #[arg(short, long, default_value_t = 12)]
    iterations: u32,

    /// Jet Partition Tolerance Factor
    #[arg(short, long, default_value_t = 0.999)]
    tolerance_factor: f64,

    /// Jet Partition Filter Ratio
    #[arg(short, long, default_value_t = 0.75)]
    filter_ratio: f64
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let graph = read_matrix_market_as_graph(Path::new(&args.mtx_filepath))?;
    let weights = gen_uniform_weights(graph.len());
    let mut partition = vec![0; graph.len()];
    let start = Instant::now();
    MultiLevelPartitioner { balance_factor: args.balance_factor,
                            num_of_partitions: args.num_of_partitions,
                            seed: None,
                            jet_iterations: args.iterations,
                            jet_tolerance_factor: args.tolerance_factor,
                            jet_filter_ratio: args.filter_ratio}.partition(&mut partition, (graph.clone(), &weights))?;
    let elapsed_time = start.elapsed();
    let edge_cut = graph.edge_cut(&partition);
    let imbalance_of_partition = imbalance(args.num_of_partitions, &partition, &weights);
    write_partition_data_to_file(&partition, &args.partition_file)?;
    println!("Edge cut {:?}", edge_cut);
    println!("Imbalance {:?}", imbalance_of_partition);
    println!("Execution time {:?}", elapsed_time);
    Ok(())
}

