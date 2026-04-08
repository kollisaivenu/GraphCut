use crate::graph::Graph;

pub struct VertexConnectivityDataStructure1 {
    conn_strength: Vec<i64>,
    num_of_parts: usize,
}

impl VertexConnectivityDataStructure1 {
    fn new(graph: &Graph, num_of_parts: usize) -> VertexConnectivityDataStructure1 {
        let conn_strength = vec![0; graph.len()*num_of_parts];

        VertexConnectivityDataStructure1 {
            conn_strength,
            num_of_parts
        }

    }
    pub fn init_vertex_connectivity_structure(graph: &Graph, num_of_parts: usize, partition: &[usize]) -> Self {
        let mut vtx_conn_data_struct = VertexConnectivityDataStructure1::new(graph, num_of_parts);

        for vertex in 0..graph.len() {

            for (neighbor_vertex, weight) in graph.neighbors(vertex) {
                vtx_conn_data_struct.increase_conn_strength(vertex, partition[neighbor_vertex], weight);
            }
        }

        vtx_conn_data_struct
    }

    pub fn get_conn_strength(&self, vertex: usize, part_id: usize) -> i64 {
        self.conn_strength[vertex*self.num_of_parts + part_id]
    }

    pub fn increase_conn_strength(&mut self, vertex: usize, part_id: usize, weight: i64) {
        self.conn_strength[self.num_of_parts*vertex + part_id] += weight;
    }

    pub fn reduce_conn_strength(&mut self, vertex: usize, part_id: usize, weight: i64) {
        self.conn_strength[self.num_of_parts*vertex + part_id] -= weight;
    }

}