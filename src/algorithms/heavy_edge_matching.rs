use rand::prelude::{SliceRandom, SmallRng};
use rustc_hash::FxHashMap;
use sprs::TriMat;
use crate::graph::Graph;

// This function coarsens the graph using heavy edge matching algorithm.
pub(crate) fn heavy_edge_matching_coarse(graph: &Graph, rng: &mut SmallRng, weights: &[i64]) -> (Graph, Vec<usize>, Vec<i64>) {

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

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
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
}