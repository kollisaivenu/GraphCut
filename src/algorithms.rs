// This file has code from https://github.com/LIHPC-Computational-Geometry/coupe
use std::fmt;

mod jet_refiner;
mod recursive_bisection;
mod multilevel_partitioner;

use jet_refiner::JetRefiner;
use recursive_bisection::Rcb;
use recursive_bisection::Point2D;
pub use multilevel_partitioner::MultiLevelPartitioner;


/// Common errors thrown by algorithms.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Error {
    /// No partition that matches the given criteria could been found.
    NotFound,

    /// Input sets don't have matching lengths.
    InputLenMismatch { expected: usize, actual: usize },

    /// Input contains negative values and such values are not supported.
    NegativeValues,

    /// When a partition improving algorithm is given more than 2 parts.
    BiPartitioningOnly,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotFound => write!(f, "no partition found"),
            Error::InputLenMismatch { expected, actual } => write!(
                f,
                "input sets don't have the same length (expected {expected} items, got {actual})",
            ),
            Error::NegativeValues => write!(f, "input contains negative values"),
            Error::BiPartitioningOnly => write!(f, "expected no more than two parts"),
        }
    }
}

impl std::error::Error for Error {}



