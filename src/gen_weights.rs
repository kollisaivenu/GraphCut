use rand::{Rng};

/// Generate the weight vector where each vertice has the same weight
pub fn gen_uniform_weights(no_of_vertices: usize) -> Vec<i64> {
    vec![1; no_of_vertices]
}

/// Generate the weight vector where each vertice has a random weight
pub fn gen_random_weights(no_of_vertices: usize, min_weight: i64, max_weight: i64) -> Vec<i64> {
    if(max_weight < min_weight) {
        panic!("Max weight must be greater than min weight.");
    }

    if(max_weight < 0 || min_weight <= 0) {
        panic!("Max/min weight must be non-negative.");
    }
    let mut rng = rand::thread_rng();

    let random_weights: Vec<i64> = (0..no_of_vertices)
        .map(|_| rng.gen_range(min_weight..max_weight))
        .collect();

    random_weights
}
