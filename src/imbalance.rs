// This file has code from https://github.com/LIHPC-Computational-Geometry/coupe
use itertools::Itertools;
use num_traits::FromPrimitive;
use num_traits::ToPrimitive;
use num_traits::Zero;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Sub;

// Calculates the total weight for each part of a given partition.
pub fn compute_parts_load<W>(partition: &[usize], num_parts: usize, weights: W) -> Vec<W::Item>
where
    W: IntoParallelIterator,
    W::Iter: IndexedParallelIterator,
    W::Item: Zero + Clone + AddAssign,
{
    debug_assert!(*partition.par_iter().max().unwrap_or(&0) < num_parts);

    partition
        .par_iter()
        .zip(weights)
        .fold(
            || vec![W::Item::zero(); num_parts],
            |mut acc, (&part, w)| {
                acc[part] += w;
                acc
            },
        )
        .reduce_with(|mut weights0, weights1| {
            for (w0, w1) in weights0.iter_mut().zip(weights1) {
                *w0 += w1;
            }
            weights0
        })
        .unwrap_or_else(|| vec![W::Item::zero(); num_parts])
}

/// Compute the imbalance of the given partition.
pub fn imbalance<W>(num_parts: usize, partition: &[usize], weights: W) -> f64
where
    W: IntoParallelIterator,
    W::Iter: IndexedParallelIterator,
    W::Item: Clone + PartialOrd + PartialEq,
    W::Item: Zero + FromPrimitive + ToPrimitive,
    W::Item: AddAssign + Div<Output = W::Item> + Sub<Output = W::Item> + Sum,
{
    let weights = weights.into_par_iter();
    debug_assert_eq!(partition.len(), weights.len());

    if num_parts == 0 {
        // Avoid a division by zero.
        return 0.0;
    }

    let part_loads = compute_parts_load(partition, num_parts, weights);
    let total_weight: W::Item = part_loads.iter().cloned().sum();

    let ideal_part_weight = total_weight.to_f64().unwrap() / num_parts.to_f64().unwrap();
    if ideal_part_weight == 0.0 {
        // Avoid divisions by zero.
        return 0.0;
    }

    part_loads
        .into_iter()
        .map(|part_weight| {
            let part_weight: f64 = part_weight.to_f64().unwrap();
            (part_weight - ideal_part_weight) / ideal_part_weight
        })
        .minmax()
        .into_option()
        .unwrap_or((0.0, 0.0))
        .1
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use itertools::assert_equal;
    use crate::imbalance::{compute_parts_load, imbalance};

    #[test]
    fn test_compute_parts_load() {
        // Arrange
        let partition = [0, 0, 1, 1];
        let vtx_weights = [4.0, 7.0, 5.0, 2.0];
        let num_parts = 2;

        // Act
        let partition_weights = compute_parts_load(&partition, num_parts, vtx_weights);

        // Assert
        assert_equal(partition_weights, [11.0, 7.0]);
    }

    #[test]
    fn test_imbalance() {
        // Arrange
        let partition = [0, 0, 1, 1];
        let vtx_weights = [3.0, 3.0, 2.0, 2.0];
        let num_parts = 2;

        // Act
        let imb = imbalance(num_parts, &partition, vtx_weights);

        // Assert
        assert_ulps_eq!(imb, 0.2);
    }
}