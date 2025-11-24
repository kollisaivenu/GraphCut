// This file has code from https://github.com/LIHPC-Computational-Geometry/coupe
pub mod graph;
pub mod gen_weights;
pub mod imbalance;
pub mod io;
pub mod algorithms;

// The `Partition` trait allows for partitioning data.
// Partitioning algorithms implement this trait.
// The generic argument `M` defines the input of the algorithms (e.g. an
// adjacency matrix or a 2D set of points).
// The input partition must be of the correct size and its contents may or may
// not be used by the algorithms.
pub trait Partition<M> {
    // Diagnostic data returned for a specific run of the algorithm.
    type Metadata;

    // Error details, should the algorithm fail to run.
    type Error;

    // Partition the given data and output the part ID of each element in
    // `part_ids`.
    //
    // Part IDs must be contiguous and start from zero, meaning the number of
    // parts is one plus the maximum of `part_ids`.  If a lower ID does not
    // appear in the array, the part is assumed to be empty.
    fn partition(&mut self, part_ids: &mut [usize], data: M)
                 -> Result<Self::Metadata, Self::Error>;
}
