use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use crate::algorithms::{Error, JetRefiner, Greedy};
use crate::algorithms::heavy_edge_matching::heavy_edge_matching_coarse;
use crate::Partition;
use crate::graph::Graph;

fn multilevel_partitioner(
    partition: &mut [usize],
    weights: &[i64],
    graph: Graph,
    num_of_partitions: usize,
    seed: Option<u64>,
    jet_iterations: u32,
    balance_factor: f64,
    jet_filter_ratio: f64,
    jet_tolerance_factor: f64,

) {

    let mut coarse_graphs = Vec::new();
    coarse_graphs.push(graph.clone());
    let mut fine_vertex_to_coarse_vertex_mappings = Vec::new();
    let mut weights_coarse_graphs = Vec::new();
    weights_coarse_graphs.push(weights.to_vec());

    let mut rng = match seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_entropy()
    };

    // Keep coarsening the graph until the graph has less than 100 nodes
    while coarse_graphs.last().unwrap().len() > num_of_partitions.max(100)  {

        let (coarse_graph, fine_vertex_to_coarse_vertex_mapping, weights_of_coarse_graph) = heavy_edge_matching_coarse(coarse_graphs.last().unwrap(), &mut rng, weights_coarse_graphs.last().unwrap());
        // Store the coarse graphs at every level
        coarse_graphs.push(coarse_graph);
        // Store the node weights of every coarse graph at each level
        weights_coarse_graphs.push(weights_of_coarse_graph);
        // Store the vertex mapping (coarse node to finer nodes) of the coarse graph at each level.
        fine_vertex_to_coarse_vertex_mappings.push(fine_vertex_to_coarse_vertex_mapping);

    }

    //let mut coarse_graph_partition = vec![0; coarse_graph_after_operation.len()];
    let mut coarse_graph_partition = vec![0; coarse_graphs.last().unwrap().len()];

    if cfg!(test){
        let mut rng = SmallRng::seed_from_u64(5);
        coarse_graph_partition.iter_mut().for_each(|vtx_partition| {
            // Generate a random integer in the half-open range [0, 2).
            // This exclusively produces either 0 or 1.
            *vtx_partition = rng.gen_range(0..2) as usize
        });
    } else {
        Greedy { part_count: num_of_partitions }.partition(&mut coarse_graph_partition, weights_coarse_graphs.last().unwrap().clone()).unwrap();
    }

    let mut index = coarse_graphs.len() - 1;

    while index >= 0 {

        // Run Jet Refiner to improve the partition.
        JetRefiner {
            num_of_partitions,
            iterations: jet_iterations,
            tolerance_factor: jet_tolerance_factor,
            balance_factor,
            filter_ratio: jet_filter_ratio}.partition(&mut coarse_graph_partition,
                                                      (coarse_graphs[index].clone(),
                                                       &weights_coarse_graphs[index])).unwrap();

        // Uncoarsen the graph till we reach the initial graph.
        if index > 0 {
            coarse_graph_partition = partition_uncoarse(&coarse_graph_partition, &fine_vertex_to_coarse_vertex_mappings[index-1]);
        } else {
            break;
        }

        index -= 1;
    }

    // Copy over the final partition to the partition array which is passed as input.
    partition.copy_from_slice(&coarse_graph_partition);
}

// Refines the partition from a coarse graph back to the original finer graph.
fn partition_uncoarse(partition: &[usize], fine_vertex_to_coarse_vertex_mapping: &Vec<usize>) -> Vec<usize>{
    // Calculate the number of vertices in the uncoarsed graph (1 up level)


    // Create a partition array for the uncoarsed graph (1 level up)
    // If vertex 1 and 2 of the uncoarsed graph were merged into vertex 0 in the coarsed graph
    // and it belonged to partition 0, then vertex 1 and 2 would belong to partition 0 in the uncoarsed graph.
    let mut new_partition: Vec<usize> = vec![0; fine_vertex_to_coarse_vertex_mapping.len()];

    for vertex in 0..fine_vertex_to_coarse_vertex_mapping.len(){
        new_partition[vertex] = partition[fine_vertex_to_coarse_vertex_mapping[vertex]];
    }

    new_partition
}

/// Multilevel Partitioner
///
/// An implementation of the Multilevel (Heavy Edge Matching && (Recursive Coordinate Bisection || Geometric Partitioner))
/// Partitioner algorithm for graph partition.
///
/// # Example
///
/// ```rust
/// use std::path::Path;
/// use GraphCut::io::{read_matrix_market_as_graph, write_partition_data_to_file};
/// use GraphCut::gen_weights::gen_random_weights;
/// use GraphCut::algorithms::MultiLevelPartitioner;
/// use GraphCut::imbalance::imbalance;
/// use GraphCut::Partition;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
///
///     let graph = read_matrix_market_as_graph(Path::new("./testdata/vt2010.mtx"))?;
///     let weights = gen_random_weights(graph.len(), 1, 3);
///     let mut partition = vec![0; graph.len()];
///
///     MultiLevelPartitioner {..Default::default()}.partition(&mut partition, (graph.clone(), &weights))?;
///
///     let edge_cut = graph.edge_cut(&partition);
///     Ok(())
/// }
/// ```

#[derive(Debug, Clone, Copy)]
pub struct MultiLevelPartitioner {
    /// Number of partitions
    pub num_of_partitions: usize,

    /// Seed for MultiLevel Graph Partitioner
    pub seed: Option<u64>,

    /// This indicates the number of times jetlp/jetrw combination should run without seeing
    /// any improvement before terminating the algorithm
    pub jet_iterations: u32,

    /// A numerical factor ranging between 0.0 and 1.0 that determines the maximum allowable
    /// deviation for a partition. The maximum weight of a partition with a balance factor of lambda
    /// can be (1+lambda)*((totol weight of graph)/(number of partitions)).
    pub balance_factor: f64,

    /// A numerical ratio ranging from 0.0 to 1.0 that determines which vertices are eligible for consideration based on
    /// their gain value in the first filter. A vertice would be considered
    /// if -gain(vertice) > (filter ratio)*(maximum connectivity of the vertice to any destination partition)
    pub jet_filter_ratio: f64,

    /// A nymerical factor ranging from 0.0 to 1.0 that is used to determine when to reset the iteration counter.
    /// If the new edge cut is less than tolerance factor times the best edge cut, then the
    /// iteration counter would be reset, otherwise the iteration counter would increment
    /// as it indicates the edge is becoming better at a very slow pace.
    pub jet_tolerance_factor: f64,

}

impl Default for MultiLevelPartitioner {
    fn default() -> Self {
        MultiLevelPartitioner {
            num_of_partitions: 2,
            seed: None,
            jet_iterations: 12,
            balance_factor: 0.1,
            jet_filter_ratio: 0.75,
            jet_tolerance_factor: 0.99,
        }
    }
}

impl<'a> Partition<(Graph, &'a [i64])> for MultiLevelPartitioner {
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (Graph, &'a [i64]),
    ) -> Result<Self::Metadata, Self::Error> {

        if part_ids.len() != weights.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: weights.len(),
            });
        }
        if part_ids.len() != adjacency.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: adjacency.len(),
            });
        }
        let metadata = multilevel_partitioner(
            part_ids,
            weights,
            adjacency,
            self.num_of_partitions,
            self.seed,
            self.jet_iterations,
            self.balance_factor,
            self.jet_filter_ratio,
            self.jet_tolerance_factor,
        );
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use crate::gen_weights::gen_uniform_weights;
    use crate::imbalance::imbalance;
    use crate::io::read_matrix_market_as_graph;
    use super::*;

    #[test]
    fn test_partition_uncoarse() {
        // Arrange
        let fine_vertex_to_coarse_vertex_mapping = vec![0, 2, 1, 0];
        let weights_coarse_graph = [5, 7, 6];
        let coarse_graph_partition = [1, 0, 0];
        let weights_uncoarse_graph = [2, 6, 7, 3];

        // Act
        let uncoarsed_graph_partition = partition_uncoarse(&coarse_graph_partition, &fine_vertex_to_coarse_vertex_mapping);

        // Assert
        assert_eq!(uncoarsed_graph_partition, vec![1, 0, 0, 1]);
        let epsilon = 1e9;
        let coarse_graph_imbalance = imbalance(2, &coarse_graph_partition, &weights_coarse_graph);
        let uncoarse_graph_imbalance = imbalance(2, &uncoarsed_graph_partition, &weights_uncoarse_graph);
        assert!((coarse_graph_imbalance - uncoarse_graph_imbalance).abs() < epsilon);
    }

    #[test]
    fn test_multilevel_partitioner_on_vt2010() {
        let graph = read_matrix_market_as_graph(Path::new("./testdata/vt2010.mtx")).unwrap();
        let weights = gen_uniform_weights(graph.len());
        let seed = Some(5);
        let mut partition = vec![0; graph.len()];
        multilevel_partitioner(&mut partition, &weights, graph.clone(), 2, seed, 12, 0.1, 0.75, 0.99);
        assert_eq!(graph.edge_cut(&partition), 15623509);
    }
}