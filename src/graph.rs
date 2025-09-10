// This file has code from https://github.com/LIHPC-Computational-Geometry/coupe

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator as _;
use std::iter::{Cloned, Zip};
use std::slice::Iter;
use ::sprs::CsMat;

/// Struct that represents a graph
pub struct Graph{
    /// The CsMat (from sprs) is used to store the graph as a sparse matrix in CSR format
    pub graph_csr: CsMat<f64>
}

impl Graph {

    /// Create a new graph
    pub fn new() -> Self {
        Self {
            graph_csr: CsMat::empty(sprs::CSR, 0)
        }
    }

    /// The number of vertices in the graph.
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.graph_csr.rows(), self.graph_csr.cols());
        self.graph_csr.rows()
    }

    /// Whether the graph has no vertices.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// An iterator over the neighbors of the given vertex.
    pub fn neighbors(&self, vertex: usize) -> Zip<Cloned<Iter<'_, usize>>, Cloned<Iter<'_, f64>>> {
        let (indices, data) = self.graph_csr.outer_view(vertex).unwrap().into_raw_storage();
        indices.iter().cloned().zip(data.iter().cloned())
    }

    /// Insert an edge with two vertices on either ends.
    pub fn insert(&mut self, vertex1: usize, vertex2: usize, edge_weight: f64) {
        self.graph_csr.insert(vertex1, vertex2, edge_weight);
    }

    /// Get edge weight for a pair of vertices.
    pub fn get_edge_weight(&self, vertex1: usize, vertex2: usize) -> Option<f64> {
        self.graph_csr.get(vertex1, vertex2).cloned()
    }

    /// The edge cut of a partition.
    ///
    /// Given a partition and a weighted graph associated to a mesh, the edge
    /// cut of a partition is defined as the total weight of the edges that link
    /// graph nodes of different parts.
    ///
    /// # Example
    ///
    /// A partition with two parts (0 and 1)
    /// ```text,ignore
    ///          0
    ///    1*──┆─*────* 0
    ///    ╱ ╲ ┆╱    ╱
    ///  1*  1*┆ <┈┈╱┈┈┈ Dotted line passes through edged that contribute to edge cut.
    ///    ╲ ╱ ┆   ╱     If all edges have a weight of 1 then edge_cut = 3
    ///    1*  ┆╲ ╱
    ///          * 0
    /// ```
    pub fn edge_cut(&self, partition: &[usize]) -> f64
    {
        debug_assert_eq!(self.len(), partition.len());

        let indptr = self.graph_csr.indptr().into_raw_storage();
        let indices = self.graph_csr.indices();
        let data = self.graph_csr.data();
        indptr
            .par_iter()
            .zip(&indptr[1..])
            .enumerate()
            .map(|(vertex, (start, end))| {
                let neighbors = &indices[*start..*end];
                let edge_weights = &data[*start..*end];
                let vertex_part = partition[vertex];
                neighbors
                    .iter()
                    .zip(edge_weights)
                    .take_while(|(neighbor, _edge_weight)| **neighbor < vertex)
                    .filter(|(neighbor, _edge_weight)| vertex_part != partition[**neighbor])
                    .map(|(_neighbor, edge_weight)| *edge_weight)
                    .sum::<f64>()
            })
            .sum()
    }

    /// Clone the graph
    pub fn clone(&self) ->Self {
        Self {
            graph_csr: self.graph_csr.clone()
        }
    }
}


