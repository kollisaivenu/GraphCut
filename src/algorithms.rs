// This file has code from https://github.com/LIHPC-Computational-Geometry/coupe
use std::fmt;

mod jet_refiner;
mod multilevel_partitioner;
mod greedy;
mod heavy_edge_matching;

use jet_refiner::JetRefiner;
use greedy::Greedy;
pub use multilevel_partitioner::MultiLevelPartitioner;


/// Common errors thrown by algorithms.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Input sets don't have matching lengths.
    InputLenMismatch { expected: usize, actual: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InputLenMismatch { expected, actual } => write!(
                f,
                "input sets don't have the same length (expected {expected} items, got {actual})",
            ),
        }
    }
}

impl std::error::Error for Error {}



