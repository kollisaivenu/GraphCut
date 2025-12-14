// This file has code from https://github.com/LIHPC-Computational-Geometry/coupe (See NOTICE.md)
use num_traits::ToPrimitive;

/// Calculates the total weight for each part of a given partition.
pub fn compute_parts_load(partition: &[usize], num_parts: usize, weights: &[i64]) -> Vec<i64> {
    let mut loads = vec![0; num_parts];

    for (&part, w) in partition.iter().zip(weights.into_iter()) {
        if part < num_parts {
            loads[part] += w;
        }
    }

    loads
}
/// Compute imbalance after passing part loads.
pub fn compute_imbalance_from_part_loads(num_parts: usize, part_loads: &Vec<i64>) -> f64 {
    let total_weight:i64 = part_loads.iter().cloned().sum();

    let ideal_part_weight = total_weight.to_f64().unwrap_or(0.0) / num_parts.to_f64().unwrap_or(1.0);
    if ideal_part_weight == 0.0 {
        return 0.0;
    }

    let max_deviation = part_loads
        .into_iter()
        .map(|part_weight| {
            let part_weight: f64 = part_weight.to_f64().unwrap_or(0.0);
            (part_weight - ideal_part_weight) / ideal_part_weight
        })
        .fold(0.0f64, |acc, dev| acc.max(dev));

    max_deviation
}
/// Compute the imbalance of the given partition.
pub fn imbalance(num_parts: usize, partition: &[usize], weights: &[i64]) -> f64 {
    if num_parts == 0 {
        return 0.0;
    }

    let part_loads = compute_parts_load(partition, num_parts, weights);

    compute_imbalance_from_part_loads(num_parts, &part_loads)
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
        let vtx_weights = vec![4, 7, 5, 2];
        let num_parts = 2;

        // Act
        let partition_weights = compute_parts_load(&partition, num_parts, &vtx_weights);

        // Assert
        assert_equal(partition_weights, [11, 7]);
    }

    #[test]
    fn test_imbalance() {
        // Arrange
        let partition = [0, 0, 1, 1];
        let vtx_weights = vec![3, 3, 2, 2];
        let num_parts = 2;

        // Act
        let imb = imbalance(num_parts, &partition, &vtx_weights);

        // Assert
        assert_ulps_eq!(imb, 0.2);
    }
}