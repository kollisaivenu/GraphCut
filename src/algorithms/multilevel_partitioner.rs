use std::io::{Write};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rustc_hash::FxHashMap;
use sprs::{CsMat, CsMatBase, TriMat};
use sprs::errors::StructureError;
use sprs::visu::print_nnz_pattern;
use crate::algorithms::{Error, JetRefiner, Greedy};
use crate::{Partition};
use crate::graph::Graph;
use crate::imbalance::imbalance;

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
        JetRefiner { iterations: jet_iterations,
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

// This function coarsens the graph using heavy edge matching algorithm.
fn heavy_edge_matching_coarse(graph: &Graph, rng: &mut SmallRng, weights: &[i64]) -> (Graph, Vec<usize>, Vec<i64>) {

    let mut matched_nodes = vec![0; graph.len()];
    let mut fine_vertex_to_coarse_vertex =  vec![0; graph.len()];

    let mut vertices: Vec<usize> = (0..graph.len()).collect();
    vertices.shuffle(rng);
    let mut super_vertex = 0usize;
    let mut num_of_edges = graph.graph_csr.nnz();

    // Iterate over the vertices of the graph.
    for vertex in vertices{
        // If already matched, then ignore
        if matched_nodes[vertex] == 1 {
            continue;
        }
        // For each vertice, finds its most connected vertice, i.e the vertice that
        // is connected with the greatest edge weight
        let mut heaviest_edge_weight = 0;
        let mut heaviest_edge_connected_vertice = None;

        for (neighbor_vertex, edge_weight) in graph.neighbors(vertex){
            // Ensure the most connected vertice is not already matched.
            if edge_weight > heaviest_edge_weight && !(matched_nodes[neighbor_vertex] == 1) {
                heaviest_edge_weight = edge_weight;
                heaviest_edge_connected_vertice = Some(neighbor_vertex);
            }
        }

        if !heaviest_edge_connected_vertice.is_none() {
            // The original node and its most connected vertex are now considered matched.
            matched_nodes[vertex] = 1;
            matched_nodes[heaviest_edge_connected_vertice.unwrap()] = 1;

            // Map the original vertex to its vertex in the coarse graph
            // This will come in handy during the reconstruction of the coarse graph.
            fine_vertex_to_coarse_vertex[vertex] = super_vertex;
            fine_vertex_to_coarse_vertex[heaviest_edge_connected_vertice.unwrap()] = super_vertex;
            num_of_edges -= 1;
        } else {
            matched_nodes[vertex] = 1;
            fine_vertex_to_coarse_vertex[vertex] = super_vertex;
        }
        super_vertex += 1;
    }

    //  We combine the edges of a vertex whose neighbors are merged in the coarsed graph.
    // Eg. If vertex 0 is connected to vertex 2 and vertex 3 which is merged into vertex 1 in the
    // coarse graph, then in the coarse graph vertex 0 will be connected to vertex 1 with
    // an edge length that is tge sum of vertex 0 and vertex 2 and vertex 0 and vertex 3
    let mut edge_to_weight_mapping = FxHashMap::with_capacity_and_hasher(num_of_edges, Default::default());

    for vertex in 0..graph.len() {
        for (neighbor, edge_weight) in graph.neighbors(vertex){

            if fine_vertex_to_coarse_vertex[vertex] != fine_vertex_to_coarse_vertex[neighbor] {
                let key = (fine_vertex_to_coarse_vertex[vertex], fine_vertex_to_coarse_vertex[neighbor]);
                let total_edge_weight = edge_to_weight_mapping.entry(key).or_insert(0);
                *total_edge_weight += edge_weight;
            }
        }
    }

    // Construction of the coarse graph. First contruct a TriMat and then convert it to CSR format.
    // This is more efficient.
    let mut new_coarse_graph  = Graph::new();
    let mut triplet_matrix = TriMat::with_capacity((super_vertex, super_vertex), num_of_edges);

    for (&(vertex1, vertex2), &weight) in edge_to_weight_mapping.iter(){
       triplet_matrix.add_triplet(vertex1, vertex2, weight);
    }

    new_coarse_graph.graph_csr = triplet_matrix.to_csr();

    // Construction of the weights array for the coarse graph.
    let mut weights_coarse_graph = vec![0; new_coarse_graph.len()];

    // Determine the new weights of the vertices.
    for vertex in 0..fine_vertex_to_coarse_vertex.len(){
        weights_coarse_graph[fine_vertex_to_coarse_vertex[vertex]] += weights[vertex];
    }

    (new_coarse_graph, fine_vertex_to_coarse_vertex, weights_coarse_graph)
}

// Refines the partition from a coarse graph back to the original finer graph.
fn partition_uncoarse(partition: &[usize], fine_vertex_to_coarse_vertex_mapping: &Vec<usize>) -> Vec<usize>{
    // Calculate the number of vertices in the uncoarsed graph (1 up level)


    // Create a partition array for the uncoarsed graph (1 level up)
    // If vertex 1 and 2 of the uncoarsed graph were merged into vertex 0 in the coarsed graph
    // and it belonged to partition 0, then vertex 1 and 2 would belong to partition 0 in the uncoarsed graph.
    let mut new_partition: Vec<usize> = vec![0; fine_vertex_to_coarse_vertex_mapping.len()];

    for vertex in (0..fine_vertex_to_coarse_vertex_mapping.len()){
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
///     let graph = read_matrix_market_as_graph(Path::new("./testdata/vt2010.mtx"));
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
    use crate::gen_weights::{gen_uniform_weights};
    use crate::io::read_matrix_market_as_graph;
    use super::*;

    #[test]
    fn test_3_node_heavy_edge_matching_coarse() {
        // Arrange
        let mut graph = Graph::new();
        graph.insert(0, 1, 5);
        graph.insert(0, 2, 10);
        graph.insert(1, 2, 15);

        graph.insert(1, 0, 5);
        graph.insert(2, 0, 10);
        graph.insert(2, 1, 15);

        let weights = [3, 4, 5];
        let mut rng = SmallRng::seed_from_u64(5);

        // Act
        let (coarse_graph, fine_vertex_to_coarse_vertex_mapping, weights_coarse_graph) = heavy_edge_matching_coarse(&graph, &mut rng, &weights);

        // Assert
        assert_eq!(20, coarse_graph.get_edge_weight(0, 1).unwrap());
        assert_eq!(20, coarse_graph.get_edge_weight(1, 0).unwrap());

        assert!(coarse_graph.get_edge_weight(0, 0).is_none());
        assert!(coarse_graph.get_edge_weight(1, 1).is_none());

        assert_eq!(fine_vertex_to_coarse_vertex_mapping, vec![0, 1, 0]);

        assert_eq!(weights_coarse_graph, vec![8, 4]);
    }

    #[test]
    fn test_5_node_heavy_edge_matching_coarse() {
        // Arrange
        let mut graph = Graph::new();
        graph.insert(0, 1, 3);
        graph.insert(1, 2, 5);
        graph.insert(2, 3, 4);
        graph.insert(3, 4, 6);
        graph.insert(4, 0, 10);

        graph.insert(1, 0, 3);
        graph.insert(2, 1, 5);
        graph.insert(3, 2, 4);
        graph.insert(4, 3, 6);
        graph.insert(0, 4, 10);

        let mut rng = SmallRng::seed_from_u64(5);
        let weights = [1, 2, 3, 4, 5];

        // Act
        let (coarse_graph, fine_vertex_to_coarse_vertex_mapping, weights_coarse_graph) = heavy_edge_matching_coarse(&graph, &mut rng, &weights);

        // Assert
        assert_eq!(3, coarse_graph.get_edge_weight(0, 1).unwrap());
        assert_eq!(3, coarse_graph.get_edge_weight(1, 0).unwrap());

        assert_eq!(6, coarse_graph.get_edge_weight(0, 2).unwrap());
        assert_eq!(6, coarse_graph.get_edge_weight(2, 0).unwrap());

        assert_eq!(4, coarse_graph.get_edge_weight(1, 2).unwrap());
        assert_eq!(4, coarse_graph.get_edge_weight(2, 1).unwrap());

        assert!(coarse_graph.get_edge_weight(0, 0).is_none());
        assert!(coarse_graph.get_edge_weight(1, 1).is_none());
        assert!(coarse_graph.get_edge_weight(2, 2).is_none());

        assert_eq!(fine_vertex_to_coarse_vertex_mapping, vec![0, 1, 1, 2, 0]);

        assert_eq!(weights_coarse_graph, vec![6, 5, 4]);
    }

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
        let graph = read_matrix_market_as_graph(Path::new("./testdata/vt2010.mtx"));
        let weights = gen_uniform_weights(graph.len());
        let seed = Some(5);
        let mut partition = vec![0; graph.len()];
        multilevel_partitioner(&mut partition, &weights, graph.clone(), 2, seed, 12, 0.1, 0.75, 0.99);
        assert_eq!(graph.edge_cut(&partition), 15623509);
    }
}