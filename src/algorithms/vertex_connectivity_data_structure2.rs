use crate::graph::Graph;

pub struct VertexConnectivityDataStructure2 {
    conn_strength: Vec<i64>,
    conn_parts: Vec<i32>,
    start_index: Vec<usize>,
    table_sizes: Vec<usize>,
    num_of_parts: usize,
}

impl VertexConnectivityDataStructure2 {
    fn new(graph: &Graph, num_of_parts: usize) -> VertexConnectivityDataStructure2 {
        let num_of_vertices = graph.len();
        let mut table_sizes = vec![0; num_of_vertices];
        let mut start_index = vec![0; num_of_vertices];
        let mut start = 0;
        for vertex in 0..num_of_vertices {
            start_index[vertex] = start;
            table_sizes[vertex] = std::cmp::min(graph.neighbors(vertex).count(), num_of_parts);
            start += table_sizes[vertex];
        }
        let mut conn_strength = vec![0; start];
        let mut conn_parts = vec![-1; start];

        VertexConnectivityDataStructure2 {
            table_sizes,
            start_index,
            conn_parts,
            conn_strength,
            num_of_parts
        }
    }

    pub fn init_vertex_connectivity_structure(graph: &Graph, num_of_parts: usize, partition: &[usize]) -> Self {
        let mut vertex_conn_data_structure = VertexConnectivityDataStructure2::new(graph, num_of_parts);

        for vertex in 0..graph.len() {
            for (neighbor_vertex, weight) in graph.neighbors(vertex) {
                vertex_conn_data_structure.increase_conn_strength(vertex, partition[neighbor_vertex], weight);
            }
        }
        vertex_conn_data_structure
    }


    pub fn get_conn_strength(&self, vertex: usize, part_id: usize) -> i64 {
        let start = self.start_index[vertex];
        let size = self.table_sizes[vertex];

        for i in 0..size {
            let index = start +  (part_id + i) % size;
            let part = self.conn_parts[index];
            if part == part_id as i32 {
                return self.conn_strength[index];
            }
        }
        0
    }

    pub fn increase_conn_strength(&mut self, vertex: usize, part_id: usize, weight: i64) {
        let start = self.start_index[vertex];
        let size = self.table_sizes[vertex];
        let mut possible_entry = None;
        for i in 0..size {
            let index = start + (part_id + i) % size;
            if self.conn_parts[index] == part_id as i32 {
                self.conn_strength[index] += weight;
                return;
            } else if self.conn_parts[index] == -1 && possible_entry.is_none() {
                possible_entry = Some(index);

            }
        }

        self.conn_parts[possible_entry.unwrap()] = part_id as i32;
        self.conn_strength[possible_entry.unwrap()] = weight;
    }

    pub fn reduce_conn_strength(&mut self, vertex: usize, part_id: usize, weight: i64) {
        let start = self.start_index[vertex];
        let size = self.table_sizes[vertex];

        for i in 0..size {
            let index = start + (part_id + i) % size;
            if self.conn_parts[index] == part_id as i32{
                self.conn_strength[index] -= weight;

                if self.conn_strength[index] == 0 {
                    self.conn_parts[index] = -1;
                }
                return;
            }
        }
    }
}
