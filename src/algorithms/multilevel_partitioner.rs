use std::collections::{HashMap, HashSet};
use std::io::{Write};
use rand::seq::SliceRandom;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use crate::algorithms::{JetRefiner, Rcb, Error, Point2D};
use crate::{Partition};
use crate::graph::Graph;
use crate::imbalance::imbalance;

fn multilevel_partitioner(
    partition: &mut [usize],
    weights: &[f64],
    graph: Graph,
    fa2_iterations: u32,
    jet_iterations: u32,
    balance_factor: f64,
    jet_filter_ratio: f64,
    jet_tolerance_factor: f64,

) {
    let mut coarse_graph_after_operation = graph.clone();
    let mut coarse_graphs = Vec::new();
    let mut vertex_mappings = Vec::new();
    let mut weights_coarse_graphs = Vec::new();

    let mut weights_of_coarse_graph_after_operation = weights.to_vec();

    // Keep coarsening the graph until the graph has less than 100 nodes
    while coarse_graph_after_operation.len() > 100  {

        let (coarse_graph, vertex_mapping, weights_of_coarse_graph) = heavy_edge_matching_coarse(&coarse_graph_after_operation, None, &weights_of_coarse_graph_after_operation);
        coarse_graph_after_operation = coarse_graph.clone();
        // Store the coarse graphs at every level
        coarse_graphs.push(coarse_graph);

        weights_of_coarse_graph_after_operation = weights_of_coarse_graph.clone();
        // Store the node weights of every coarse graph at each level
        weights_coarse_graphs.push(weights_of_coarse_graph);
        // Store the vertex mapping (coarse node to finer nodes) of the coarse graph at each level.
        vertex_mappings.push(vertex_mapping);

    }

    let mut coarse_graph_partition = vec![0; coarse_graph_after_operation.len()];
    // Use forceatlas2 algorithm to provide co-ordinates for each node in the graph to perform the initial partitions
    let points = convert_graph_to_coordinates(&coarse_graph_after_operation, weights_of_coarse_graph_after_operation.clone(), fa2_iterations);

    // Use Recursive Initial Bisection for initial partition.
    Rcb { iter_count: 1, tolerance: balance_factor}.partition(&mut coarse_graph_partition, (points, weights_of_coarse_graph_after_operation.clone())).unwrap();

    let mut index = coarse_graphs.len() - 2;

    while index >= 0 {
        // Uncoarsen the graph till we reach the initial graph.
        coarse_graph_partition = partition_uncoarse(&coarse_graph_partition, &vertex_mappings[index+1]);
        // Run Jet Refiner to improve the partition.
        JetRefiner { iterations: jet_iterations, 
            tolerance_factor: jet_tolerance_factor, 
            balance_factor: balance_factor, 
            filter_ratio: jet_filter_ratio}.partition(&mut coarse_graph_partition,
                                                      (coarse_graphs[index].clone(),
                                                       &weights_coarse_graphs[index])).unwrap();

        if index == 0 {
            break;
        }
        index -= 1;
    }
    let final_graph_partition = partition_uncoarse(&coarse_graph_partition, &vertex_mappings[0]);
    // Copy over the final partition to the partition array which is passed as input.
    partition.copy_from_slice(&final_graph_partition);
}

// This function coarsens the graph using heavy edge matching algorithm.
fn heavy_edge_matching_coarse(graph: &Graph, seed: Option<u64>, weights: &[f64]) -> (Graph, Vec<Vec<usize>>, Vec<f64>) {

    let mut rng = match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy()
    };

    let mut matched_nodes: HashSet<usize> = HashSet::new();
    let mut vertex_mapping = Vec::new();
    let mut old_vertex_to_new_vertex =  HashMap::new();

    let mut vertices: Vec<usize> = (0..graph.len()).collect();
    vertices.shuffle(&mut rng);

    let mut super_vertex = 0usize;
    // Iterate over the vertices of the graph.
    for vertex in vertices{
        // If already matched, then ignore
        if matched_nodes.contains(&vertex){
            continue;
        }
        // For each vertice, finds its most connected vertice, i.e the vertice that
        // is connected with the greatest edge weight
        let mut heaviest_edge_weight = 0f64;
        let mut heaviest_edge_connected_vertice = None;

        for (neighbor_vertex, edge_weight) in graph.neighbors(vertex){
            // Ensure the most connected vertice is not already matched.
            if edge_weight > heaviest_edge_weight && !matched_nodes.contains(&neighbor_vertex) {
                heaviest_edge_weight = edge_weight;
                heaviest_edge_connected_vertice = Some(neighbor_vertex);
            }
        }

        if !heaviest_edge_connected_vertice.is_none() {
            // After determining the most connected vertice, we merge the original vertice and
            // its most connected vertice into "supervertex"
            vertex_mapping.push(vec![vertex.min(heaviest_edge_connected_vertice.unwrap()),
                                     vertex.max(heaviest_edge_connected_vertice.unwrap())]);
            // The original node and its most connected vertex are now considered matched.
            matched_nodes.insert(vertex);
            matched_nodes.insert(heaviest_edge_connected_vertice.unwrap());

            // Map the original vertex to its vertex in the coarse graph
            // This will come in handy during the reconstruction of the coarse graph.
            old_vertex_to_new_vertex.insert(vertex, super_vertex);
            old_vertex_to_new_vertex.insert(heaviest_edge_connected_vertice.unwrap(), super_vertex);
        } else {
            // This flow is for the scenario when a vertex has no vertex to merge with.
            // (mostly because all of its neighbors are already matched)
            vertex_mapping.push(vec![vertex]);
            matched_nodes.insert(vertex);
            old_vertex_to_new_vertex.insert(vertex, super_vertex);
        }
        super_vertex += 1;
    }

    //  We combine the edges of a vertex whose neighbors are merged in the coarsed graph.
    // Eg. If vertex 0 is connected to vertex 2 and vertex 3 which is merged into vertex 1 in the
    // coarse graph, then in the coarse graph vertex 0 will be connected to vertex 1 with
    // an edge length that is tge sum of vertex 0 and vertex 2 and vertex 0 and vertex 3
    let mut edge_to_weight_mapping = HashMap::new();

    for vertex in 0..graph.len() {
        for (neighbor, edge_weight) in graph.neighbors(vertex){
            if old_vertex_to_new_vertex[&vertex] != old_vertex_to_new_vertex[&neighbor] {
                let key = (old_vertex_to_new_vertex[&vertex], old_vertex_to_new_vertex[&neighbor]);
                let total_edge_weight = edge_to_weight_mapping.entry(key).or_insert(0f64);
                *total_edge_weight += edge_weight;
            }
        }
    }

    // Construction of the coarse graph.
    let mut new_coarse_graph  = Graph::new();

    for key in edge_to_weight_mapping.keys(){
        let(vertex1, vertex2) = *key;
        let edge_weight = edge_to_weight_mapping.get(key).unwrap();

        new_coarse_graph.insert(vertex1, vertex2, *edge_weight);
    }

    // Construction of the weights array for the coarse graph.
    let mut weights_coarse_graph = vec![0f64; new_coarse_graph.len()];

    for coarse_vertex in 0..vertex_mapping.len(){
        for uncoarse_vertex in vertex_mapping[coarse_vertex].iter(){
            weights_coarse_graph[coarse_vertex] += weights[*uncoarse_vertex];
        }
    }

    (new_coarse_graph, vertex_mapping, weights_coarse_graph)
}

// Refines the partition from a coarse graph back to the original finer graph.
fn partition_uncoarse(partition: &[usize], vertex_mapping: &Vec<Vec<usize>>) -> Vec<usize>{
    // Calculate the number of vertices in the uncoarsed graph (1 up level)
    let mut vertices = 0;
    for mapped_vertices in vertex_mapping{
        vertices += mapped_vertices.len();
    }

    // Create a partition array for the uncoarsed graph (1 level up)
    // If vertex 1 and 2 of the uncoarsed graph were merged into vertex 0 in the coarsed graph
    // and it belonged to partition 0, then vertex 1 and 2 would belong to partition 0 in the uncoarsed graph.
    let mut new_partition: Vec<usize> = vec![0; vertices];

    for coarse_graph_vertice in 0..vertex_mapping.len(){
        let vertex_partition = partition[coarse_graph_vertice];

        for uncoarse_graph_vertice in &vertex_mapping[coarse_graph_vertice]{
            new_partition[*uncoarse_graph_vertice] = vertex_partition;
        }
    }

    new_partition
}

// This function computes 2D coordinates for graph nodes using forceatlas2 algorithm.
fn convert_graph_to_coordinates(graph: &Graph, weights: Vec<f64>, iter:u32) -> Vec<Point2D> {
    // Create a vector where the elements are in the structure ((vertex1, vertex2), edge_weight).
    let mut edges = Vec::new();

    for node in 0..graph.len() {
        for (neighbor_node, edge_weight) in graph.neighbors(node) {
            edges.push(((node, neighbor_node), edge_weight));
        }
    }

    // Run forceatlas2 for the graph to generate the coordinates.
    let mut layout = forceatlas2::Layout::<f64, 2>::from_graph_with_degree_mass(
        edges,
        weights,
        forceatlas2::Settings{strong_gravity: true , ..Default::default()},
    );

    // Run the iterations
    for _ in 0..iter {
        layout.iteration();
    }

    let mut points = Vec::with_capacity(graph.len());
    for (_, node) in layout.nodes.iter().enumerate() {
        points.push(Point2D::new(node.pos.x(), node.pos.y()));
    }

    points
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
///     let weights = gen_random_weights(graph.len(), 1.0, 3.0);
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
    /// Number of ForceAtlas2 iterations to run on the coarsed graph to get generate co-ordinates for the graph
    pub fa2_iterations: u32,

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
            fa2_iterations: 100,
            jet_iterations: 12,
            balance_factor: 0.1,
            jet_filter_ratio: 0.75,
            jet_tolerance_factor: 0.99,
        }
    }
}

impl<'a> Partition<(Graph, &'a [f64])> for MultiLevelPartitioner {
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (Graph, &'a [f64]),
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
            self.fa2_iterations,
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
    use super::*;

    #[test]
    fn test_3_node_heavy_edge_matching_coarse() {
        // Arrange
        let mut graph = Graph::new();
        graph.insert(0, 1, 5.);
        graph.insert(0, 2, 10.);
        graph.insert(1, 2, 15.);

        graph.insert(1, 0, 5.);
        graph.insert(2, 0, 10.);
        graph.insert(2, 1, 15.);

        let weights = [3.0, 4.0, 5.0];
        let seed = Some(5);

        // Act
        let (coarse_graph, vertex_mapping, weights_coarse_graph) = heavy_edge_matching_coarse(&graph, seed, &weights);


        // Assert
        assert_eq!(15., coarse_graph.get_edge_weight(0, 1).unwrap());
        assert_eq!(15., coarse_graph.get_edge_weight(1, 0).unwrap());

        assert!(coarse_graph.get_edge_weight(0, 0).is_none());
        assert!(coarse_graph.get_edge_weight(1, 1).is_none());

        assert_eq!(vertex_mapping[0], vec![1, 2]);
        assert_eq!(vertex_mapping[1], vec![0]);

        assert_eq!(weights_coarse_graph, vec![9.0, 3.0]);
    }

    #[test]
    fn test_5_node_heavy_edge_matching_coarse() {
        // Arrange
        let mut graph = Graph::new();
        graph.insert(0, 1, 3.);
        graph.insert(1, 2, 5.);
        graph.insert(2, 3, 4.);
        graph.insert(3, 4, 6.);
        graph.insert(4, 0, 10.);

        graph.insert(1, 0, 3.);
        graph.insert(2, 1, 5.);
        graph.insert(3, 2, 4.);
        graph.insert(4, 3, 6.);
        graph.insert(0, 4, 10.);

        let seed = Some(5);

        let weights = [1.0, 2.0, 3.0, 4.0, 5.0];

        // Act
        let (coarse_graph, vertex_mapping, weights_coarse_graph) = heavy_edge_matching_coarse(&graph, seed, &weights);

        // Assert
        assert_eq!(6., coarse_graph.get_edge_weight(0, 1).unwrap());
        assert_eq!(6., coarse_graph.get_edge_weight(1, 0).unwrap());

        assert_eq!(3., coarse_graph.get_edge_weight(0, 2).unwrap());
        assert_eq!(3., coarse_graph.get_edge_weight(2, 0).unwrap());

        assert_eq!(5., coarse_graph.get_edge_weight(1, 2).unwrap());
        assert_eq!(5., coarse_graph.get_edge_weight(2, 1).unwrap());

        assert!(coarse_graph.get_edge_weight(0, 0).is_none());
        assert!(coarse_graph.get_edge_weight(1, 1).is_none());
        assert!(coarse_graph.get_edge_weight(2, 2).is_none());

        assert_eq!(vertex_mapping[0], vec![0, 4]);
        assert_eq!(vertex_mapping[1], vec![2, 3]);
        assert_eq!(vertex_mapping[2], vec![1]);

        assert_eq!(weights_coarse_graph, vec![6.0, 7.0, 2.0]);
    }

    #[test]
    fn test_partition_uncoarse() {
        // Arrange
        let vertex_mapping = vec![vec![0, 3], vec![2], vec![1]];
        let weights_coarse_graph = [5.0, 7.0, 6.0];
        let coarse_graph_partition = [1, 0, 0];
        let weights_uncoarse_graph = [2.0, 6.0, 7.0, 3.0];

        // Act
        let uncoarsed_graph_partition = partition_uncoarse(&coarse_graph_partition, &vertex_mapping);

        // Assert
        assert_eq!(uncoarsed_graph_partition, vec![1, 0, 0, 1]);
        let epsilon = 1e9;
        let coarse_graph_imbalance = imbalance(2, &coarse_graph_partition, weights_coarse_graph.clone());
        let uncoarse_graph_imbalance = imbalance(2, &uncoarsed_graph_partition, weights_uncoarse_graph.clone());
        assert!((coarse_graph_imbalance - uncoarse_graph_imbalance).abs() < epsilon);
    }
}