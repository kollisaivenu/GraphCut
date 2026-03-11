// This file contains the implementation of the Jet Refiner algorithm used in the refining phase
// # Reference
//
// Gilbert, Michael S., et al. "Jet: Multilevel graph partitioning on graphics processing units."
// SIAM Journal on Scientific Computing 46.5 (2024): B700-B724.

use crate::algorithms::Error;
use crate::imbalance::{compute_imbalance_from_part_loads, compute_parts_load};
use num_traits::{ToPrimitive};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rustc_hash::{FxHashMap, FxHashSet};
use crate::graph::Graph;

#[derive(Debug)]
struct Move {
    // Struct to store data about a move that can either lead to better edge cuts or
    // re-balance the weights.

    //The index of the vertex.
    vertex: usize,

    // The partition ID of the partition where the vertex should move to.
    partition_id: usize
}

fn jet_refiner(
    partition: &mut [usize],
    weights: &[i64],
    adjacency: Graph,
    num_of_partitions: usize,
    iterations: u32,
    balance_factor: f64,
    filter_ratio: f64,
    tolerance_factor: f64,
    seed: Option<u64>
) {

    debug_assert!(!partition.is_empty());
    debug_assert_eq!(partition.len(), weights.len());
    debug_assert_eq!(partition.len(), adjacency.len());

    let mut partition_iter = partition.to_vec();
    let mut current_iteration = 0;
    let mut vertex_connectivity_data_structure = init_vertex_connectivity_data_structure(&adjacency, partition);
    let mut locked_vertices = vec![false; adjacency.len()];

    let mut partition_weights = compute_parts_load(&partition, num_of_partitions, &weights);
    let mut imbalance_of_current_iter_partition = compute_imbalance_from_part_loads(num_of_partitions, &partition_weights);
    let mut imbalance_of_best_partition = imbalance_of_current_iter_partition;
    let mut best_partition_edge_cut = adjacency.edge_cut(&partition);
    let mut curr_iter_partition_edge_cut = best_partition_edge_cut;

    let total_weight: i64 = weights.iter().cloned().sum();
    let mut random_num_gen;
    let mut dest_partititon = vec![-2; partition.len()];
    let mut gain = vec![None; partition.len()];
    let mut moves = Vec::with_capacity(partition.len());
    let mut weak_rebalance_counter = 1;

    match seed {
        Some(s) => {
            random_num_gen = SmallRng::seed_from_u64(s)
        }
        None => {
            random_num_gen = SmallRng::from_entropy()
        }
    }

    while current_iteration < iterations {

        if imbalance_of_current_iter_partition < balance_factor {
            // the jetlp subroutine is used to generate a better partition
            jetlp(&adjacency,
                  num_of_partitions,
                  &partition_iter,
                  &vertex_connectivity_data_structure,
                  &locked_vertices,
                  filter_ratio,
                  &mut dest_partititon,
                  &mut gain,
                  &mut moves);
            // Based on the suggested moves of the jetlp subroutine, the vertices to be moved are locked
            // to ensure that they don't become eligible to move in the next iteration.
            // This prevents oscillation of vertices
            lock_vertices(&moves, &mut locked_vertices);
            weak_rebalance_counter = 1;
        } else {
            // the jetrw subroutine is run to balance the weights of the partition
            // (should the partitions weights become highly imbalanced)
            // moves = jetrw(&adjacency,
            //               &partition_iter,
            //               weights,
            //               total_weight,
            //               &vertex_connectivity_data_structure,
            //               num_of_partitions,
            //               balance_factor,
            //               &mut random_num_gen,
            //               &partition_weights);

            if weak_rebalance_counter <= 2 {
                jetrw(&adjacency,
                      &partition_iter,
                      weights,
                      total_weight,
                      &vertex_connectivity_data_structure,
                      num_of_partitions,
                      balance_factor,
                      &mut random_num_gen,
                      &partition_weights,
                      &mut moves);
                weak_rebalance_counter += 1;
            } else {
                jetrs(&adjacency,
                      &partition_iter,
                      weights,
                      total_weight,
                      &vertex_connectivity_data_structure,
                      num_of_partitions,
                      balance_factor,
                      &partition_weights,
                      &mut moves);
            }
        }

        // The moves from either jetlp or jetrw are applied on the current partition state.
        update_parts_and_vertex_connectivity(&adjacency,
                                             &mut partition_iter,
                                             &mut vertex_connectivity_data_structure,
                                             &moves,
                                             &mut curr_iter_partition_edge_cut,
                                             &mut partition_weights,
                                             &weights,
                                             &mut dest_partititon,
                                             &mut gain);



        imbalance_of_current_iter_partition = compute_imbalance_from_part_loads(num_of_partitions, &partition_weights);
        moves.clear();
        // Check if the current iteration partition is balance
        if  imbalance_of_current_iter_partition < balance_factor {
            // Check if the current iteration partition is better than the current best partition
            if curr_iter_partition_edge_cut < best_partition_edge_cut {
                // Current iteration partition is chosen as the best partition
                if curr_iter_partition_edge_cut < (tolerance_factor*(best_partition_edge_cut as f64)).floor() as i64 {
                    current_iteration = 0;
                }

                partition.copy_from_slice(&partition_iter);
                imbalance_of_best_partition = imbalance_of_current_iter_partition;
                best_partition_edge_cut = curr_iter_partition_edge_cut;
            } else {
                current_iteration += 1;
            }
        } else if imbalance_of_current_iter_partition < imbalance_of_best_partition {
            // Current iteration is better balanced than the best iteration, hence this is made
            // the best iteration
            partition.copy_from_slice(&partition_iter);
            imbalance_of_best_partition = imbalance_of_current_iter_partition;
            best_partition_edge_cut = curr_iter_partition_edge_cut;
            current_iteration = 0
        } else {
            current_iteration += 1;
        }
    }
}

fn jetlp(graph: &Graph, num_of_partitions: usize, partition: &[usize], vertex_connectivity_data_structure: &[FxHashMap<usize, i64>], locked_vertices: &[bool], filter_ratio: f64, dest_partition: &mut [i64], gain: &mut [Option<i64>], moves: &mut Vec<Move>) {
    // iterate over all the vertices to find out which vertices provides the best gain (decrease in edge cut)
    for vertex in 0..graph.len() {
        // These are values if all the vertex belongs to the same partition as its neighbours.
        let mut vertex_calculated_gain = None;
        let mut vertex_dest_partition = -1;

        if !locked_vertices[vertex] && dest_partition[vertex] == -2 {

            let vertex_partition = partition[vertex];
            let mut connection_strength_dest = i64::MIN;
            let connection_strength_source = vertex_connectivity_data_structure[vertex].get(&vertex_partition).unwrap_or(&0);

            // Iterate over all partitions find out which partition is most connected to the vertex.
            for (&part, &partition_connection) in vertex_connectivity_data_structure[vertex].iter() {

                if partition_connection > 0 &&  part != vertex_partition{

                    if partition_connection > connection_strength_dest {
                        connection_strength_dest = partition_connection;
                        vertex_dest_partition = part as i64;
                    }
                }
            }

            if vertex_dest_partition >= 0  {
                vertex_calculated_gain = Some(connection_strength_dest - connection_strength_source);
            }

            dest_partition[vertex] = vertex_dest_partition;
            gain[vertex] = vertex_calculated_gain;
        }
    }

    // First filter is applied to check which of the vertices are eligible for moving from one partition
    // to another. Either the gain should be positive or can be slightly negative (based on the filter ratio).
    // Slightly negative gain vertices are also considered in the hope that they could provide better global solutions
    let first_filter_eligible_moves = gain_conn_ratio_filter(locked_vertices,
                                                                         partition,
                                                                         &gain,
                                                                         vertex_connectivity_data_structure,
                                                                         filter_ratio);

    // A heuristic attempt is made to approximate the true gain that would occur since
    // two positive moves when applied simultaneously can be detrimental.
    let first_filter_eligible_vertices = first_filter_eligible_moves.clone().into_iter().collect::<FxHashSet<_>>();

    for vertex_index in 0..first_filter_eligible_moves.len() {
        let vertex = first_filter_eligible_moves[vertex_index];
        let mut gain_for_vertex = 0;

        for (neighbor_vertex, edge_weight) in graph.neighbors(vertex){
            let mut partition_source = partition[neighbor_vertex];

            if is_higher_placed(neighbor_vertex, vertex, &gain, &first_filter_eligible_vertices) {
                partition_source = dest_partition[neighbor_vertex] as usize;
            }

            if partition_source == dest_partition[vertex] as usize {
                gain_for_vertex += edge_weight;
            } else if partition_source == partition[vertex] {
                gain_for_vertex -= edge_weight;
            }
        }

        if gain_for_vertex > 0 {
            moves.push(Move {vertex: vertex, partition_id: dest_partition[vertex] as usize});
        }
    };
}

fn jetrw(graph: &Graph, partitions: &[usize], vertex_weights: &[i64], total_weight: i64, vertex_connectivity_data_structure: &[FxHashMap<usize, i64>], num_of_partitions: usize, balance_factor: f64, random_num_gen: &mut SmallRng, partition_weights: &[i64], moves: &mut Vec<Move>) {
    // Weaker but better rebalancer in terms of the change in edgecut
    let max_slots: usize = 25;
    let max_weight_per_partitions = (1f64 + balance_factor)*(total_weight as f64)/(num_of_partitions as f64);
    let num_of_vertices = graph.len();
    let mut heavy_partitions = FxHashSet::default();
    let mut light_partitions = FxHashSet::default();
    let mut light_partitions_vec = Vec::new();

    // Set what the max weight of the destination partition can be.
    // This is to prevent oscillations when the jetrw algorithm is rerun
    let mut max_weight_dest = max_weight_per_partitions*0.99;

    if max_weight_dest < max_weight_per_partitions - 100f64 {
        max_weight_dest = max_weight_per_partitions - 100f64;
    }

    for partition_id in 0..num_of_partitions{
        let weight_of_partition = partition_weights[partition_id] as f64;

        if max_weight_per_partitions < weight_of_partition {
            heavy_partitions.insert(partition_id);
        }

        if max_weight_dest >= weight_of_partition {
            light_partitions.insert(partition_id);
            light_partitions_vec.push(partition_id);
        }
    }
    // Create a hashmap that can be used to precompute the index for every heavy partition.
    let mut heavy_partition_to_index = FxHashMap::default();
    for (index, &heavy_partition) in heavy_partitions.iter().enumerate() {
        heavy_partition_to_index.insert(heavy_partition, index);
    }

    // Find out the loss for each eligible vertex move (from an overweight partition to an underweight partition).
    // A positive loss indicates an increase in edge cut.
    let mut partitions_dest = vec![0usize; num_of_vertices];
    let mut loss = vec![0i64; num_of_vertices];

    for vertex in 0..num_of_vertices {
            let weight_of_partition = partition_weights[partitions[vertex]];

            let limit = 1.5*(weight_of_partition as f64 - ((total_weight as f64)/(num_of_partitions as f64)));

            let mut calculated_loss = 0;
            let mut dest_partition: usize = 0;

            if heavy_partitions.contains(&partitions[vertex]) && ((vertex_weights[vertex] as f64) < limit) {

                let most_connected_light_partition = get_most_connected_light_partition(vertex,
                                                                                                      &light_partitions,
                                                                                                      vertex_connectivity_data_structure);

                dest_partition = most_connected_light_partition.unwrap_or(light_partitions_vec[random_num_gen.gen_range(0..light_partitions.len())]);

                calculated_loss = conn(vertex,
                                       partitions[vertex],
                                       vertex_connectivity_data_structure) - conn(vertex,
                                                                                  dest_partition,
                                                                                  vertex_connectivity_data_structure);
            }
            partitions_dest[vertex] = dest_partition;
            loss[vertex] = calculated_loss;
        }

    // Slot the loss values into different buckets. This is to prevent sorting the loss values
    // which can be expensive.
    let mut bucket = init_bucket(heavy_partitions.len(), max_slots);

    for vertex in 0..num_of_vertices{

        if heavy_partitions.contains(&partitions[vertex]) {
            let index = heavy_partition_to_index[&partitions[vertex]];
            let slot = calculate_slot(loss[vertex], max_slots);
            bucket[get_index_for_bucket(index, slot, max_slots)].push(vertex);
        }
    }

    // For each of the heavy partitions, decide the vertices that can be moved from the
    // heavy partitions such that the increase in edge cut is minimized.
    for (index, &heavy_partition) in heavy_partitions.iter().enumerate(){
        let mut is_still_heavy_partition = true;
        let mut m = 0f64;
        let m_max = partition_weights[heavy_partition] as f64 - max_weight_per_partitions;

        for slot in 0..max_slots {

            for &vertex in &bucket[get_index_for_bucket(index, slot, max_slots)] {

                if m < m_max {
                    m = m + (vertex_weights[vertex] as f64);
                    moves.push(Move{vertex, partition_id: partitions_dest[vertex]});
                } else {
                    is_still_heavy_partition = false;
                    break;
                }
            }

            if !is_still_heavy_partition {
                break;
            }
        }
    }
}

fn jetrs(graph: &Graph, partitions: &[usize], vertex_weights: &[i64], total_weight: i64, vertex_connectivity_data_structure: &[FxHashMap<usize, i64>], num_of_partitions: usize, balance_factor: f64, partition_weights: &[i64], moves: &mut Vec<Move>) {
    // Stronger but  worse rebalancer in terms of the change in edgecut
    let max_slots: usize = 50;
    let max_weight_per_partitions = (1f64 + balance_factor)*(total_weight as f64)/(num_of_partitions as f64);
    let num_of_vertices = graph.len();
    let mut heavy_partitions = FxHashSet::default();
    let mut light_partitions = FxHashSet::default();

    // Set what the max weight of the destination partition can be.
    // This is to prevent oscillations when the jetrw algorithm is rerun
    // let mut max_weight_dest = max_weight_per_partitions*0.99;
    let opt_size = (total_weight as f64 / num_of_partitions as f64).ceil();
    let max_weight_dest = f64::max(opt_size + 1.0, max_weight_per_partitions * 0.99);
    let mut weight_to_add_to_partition = FxHashMap::default();
    for partition_id in 0..num_of_partitions{
        let weight_of_partition = partition_weights[partition_id] as f64;

        if max_weight_per_partitions < weight_of_partition {
            heavy_partitions.insert(partition_id);
        }

        if max_weight_dest >= weight_of_partition {
            light_partitions.insert(partition_id);
            weight_to_add_to_partition.insert(partition_id, max_weight_per_partitions - partition_weights[partition_id] as f64);
        }
    }


    let mut heavy_partition_to_index = FxHashMap::default();
    for (index, &heavy_partition) in heavy_partitions.iter().enumerate() {
        heavy_partition_to_index.insert(heavy_partition, index);
    }

    // Find out the avg loss for each eligible vertex move (from an overweight partition to an underweight partition).
    // A positive loss indicates an increase in edge cut.
    let loss: Vec<i64> = (0..num_of_vertices).map(|vertex| {
        let weight_of_partition = partition_weights[partitions[vertex]];
        let limit = 1.5*(weight_of_partition as f64 - ((total_weight as f64)/(num_of_partitions as f64)));
        let mut calculated_loss = 0;

        if heavy_partitions.contains(&partitions[vertex]) && ((vertex_weights[vertex] as f64) < limit) {

            let (conn_strength, no_of_light_partitions) = calculate_connection_strength_with_light_partitions(vertex,
                                                                                                              &light_partitions,
                                                                                                              vertex_connectivity_data_structure);

            calculated_loss = conn(vertex,
                                   partitions[vertex],
                                   vertex_connectivity_data_structure) - ((conn_strength as f64)/(no_of_light_partitions as f64)).floor() as i64;

        }
        calculated_loss
    }).collect();


    // Slot the loss values into different buckets. This is to prevent sorting the loss values
    // which can be expensive.
    let mut bucket = init_bucket(heavy_partitions.len(), max_slots);

    for vertex in 0..num_of_vertices{

        if heavy_partitions.contains(&partitions[vertex]) {
            let index = heavy_partition_to_index[&partitions[vertex]];
            let slot = calculate_slot(loss[vertex], max_slots);
            bucket[get_index_for_bucket(index, slot, max_slots)].push(vertex);
        }
    }

    for (index, &heavy_partition) in heavy_partitions.iter().enumerate() {
        let mut is_still_heavy = true;
        let weight_to_remove = partition_weights[heavy_partition] as f64 - max_weight_per_partitions;

        let mut weight_currently_removed = 0f64;
        for slot in 0..max_slots {
            for &vertex in &bucket[get_index_for_bucket(index, slot, max_slots)] {
                if weight_currently_removed <= weight_to_remove {
                    let mut found_light_partition = false;
                    // Move a vertex in a heavy partition to a light partition that it is connected to.
                    for (&neighboring_partition, _) in vertex_connectivity_data_structure[vertex].iter() {

                        if light_partitions.contains(&neighboring_partition) && weight_to_add_to_partition[&neighboring_partition] >= vertex_weights[vertex] as f64 {
                            weight_currently_removed = weight_currently_removed + (vertex_weights[vertex] as f64);
                            moves.push(Move{vertex, partition_id: neighboring_partition});
                            *weight_to_add_to_partition
                                .entry(neighboring_partition)
                                .or_insert(0f64) -= vertex_weights[vertex] as f64;
                            found_light_partition = true;
                            break;
                        }
                    }
                    // If we don't find a connected light partition for a vertex to move to,
                    // then we choose a random light partition to move it to
                    if !found_light_partition {
                        for &light_partition in light_partitions.iter() {
                            if weight_to_add_to_partition[&light_partition] >= vertex_weights[vertex] as f64 {
                                moves.push(Move{vertex, partition_id: light_partition});
                                *weight_to_add_to_partition
                                    .entry(light_partition)
                                    .or_insert(0f64) -= vertex_weights[vertex] as f64;
                                weight_currently_removed = weight_currently_removed + (vertex_weights[vertex] as f64);
                                break;
                            }
                        }
                    }
                } else {
                    is_still_heavy = false;
                    break;
                }
            }

            if !is_still_heavy {
                break
            }
        }
    }
}

fn calculate_connection_strength_with_light_partitions(vertex: usize, eligible_partitions: &FxHashSet<usize>, vertex_connectivity_data_structure: &[FxHashMap<usize, i64>]) -> (i64, usize) {
    // Gets the connection strength of the vertex with all the light partitions and the number of light partitions.

    let mut conn_strength = 0i64;
    let mut unique_light_partitions = FxHashSet::default();
    for (&partition_id, &strength) in &vertex_connectivity_data_structure[vertex] {
        if eligible_partitions.contains(&partition_id) {
            unique_light_partitions.insert(partition_id);
            conn_strength += strength;
        }
    }

    (conn_strength, unique_light_partitions.len())
}

fn lock_vertices(moves: &[Move], locked_vertices: &mut [bool]) {
    // This function gets the list of locked vertices that shouldn't be moved in the subsequent iterations.
    locked_vertices.fill(false);

    for single_move in moves{
        locked_vertices[single_move.vertex] = true;
    }

}

fn gain_conn_ratio_filter(locked_vertices: &[bool], partitions: &[usize], gain: &[Option<i64>], vertex_connectivity_data_structure: &[FxHashMap<usize, i64>], filter_ratio: f64) -> Vec<usize> {
    // Get a list of vertices that have a positive gain or slightly negative gain value (based on the filter ratio).

    let num_vertices = partitions.len();
    let mut list_of_moveable_vertices  = Vec::new();

    for vertex in 0..num_vertices {
        if (!locked_vertices[vertex])
            &&
            !gain[vertex].is_none()
            &&
            (gain[vertex].unwrap() > 0 || -gain[vertex].unwrap() < (filter_ratio * (conn(vertex, partitions[vertex], vertex_connectivity_data_structure) as f64)).floor() as i64){
            list_of_moveable_vertices.push(vertex);
        }
    }

    list_of_moveable_vertices
}

fn conn(vertex_id: usize,
        partition_id: usize,
        vertex_connectivity_data_structure: &[FxHashMap<usize, i64>]) ->i64 {
    // Gets how well a vertex is connected to a partition (adds all the edge weights connected to the partition).

    *vertex_connectivity_data_structure[vertex_id].get(&partition_id).unwrap_or(&0i64)
}

fn init_vertex_connectivity_data_structure(graph: &Graph,
                                           partition: &[usize]) -> Vec<FxHashMap<usize, i64>> {
    // Initialize the vertex connectivity data structure.
    let mut vertex_connectivity_data_structure = vec![FxHashMap::default(); partition.len()];

    let num_of_vertices = graph.len();

    for vertex in 0..num_of_vertices {
        let neighbours = graph.neighbors(vertex);
        for (neighbour_vertex, edge_weight) in neighbours {
            *vertex_connectivity_data_structure[vertex]
                .entry(partition[neighbour_vertex])
                .or_insert(0) += edge_weight;
        }
    }

    vertex_connectivity_data_structure
}

fn update_parts_and_vertex_connectivity(
    graph: &Graph,
    partition: &mut [usize],
    vertex_connectivity_data_structure: &mut [FxHashMap<usize, i64>],
    moves: &[Move],
    curr_iter_partition_edge_cut: &mut i64,
    partition_weights: &mut [i64],
    weights: &[i64],
    dest_partition: &mut [i64],
    gain: &mut [Option<i64>]) {
    // Updates the partitions and the vertex connectivity data structure using the given list of moves.
    for single_move in moves {
        let vertex = single_move.vertex;
        let partition_source = partition[vertex];

        *curr_iter_partition_edge_cut += vertex_connectivity_data_structure[vertex].get(&partition_source).unwrap_or(&0);
        // Setting this to -2 means it needs to be recomputed in jetlp
        dest_partition[vertex] = -2;
        gain[vertex] = None;

        for (neighbour_vertex, edge_weight) in graph.neighbors(vertex) {
            *vertex_connectivity_data_structure[neighbour_vertex].entry(partition_source).or_insert(0) -= edge_weight;
            // Setting this to -2 means it needs to be recomputed in jetlp
            if vertex_connectivity_data_structure[neighbour_vertex][&partition_source] == 0 {
                vertex_connectivity_data_structure[neighbour_vertex].remove(&partition_source);
            }
            dest_partition[neighbour_vertex] = -2;
            gain[neighbour_vertex] = None;
        }

        partition_weights[partition_source] -= weights[vertex];
        partition_weights[single_move.partition_id] += weights[vertex];
        partition[vertex] = single_move.partition_id;
    }

    for single_move in moves {
        let vertex = single_move.vertex;
        let partition_dest = single_move.partition_id;

        *curr_iter_partition_edge_cut -= vertex_connectivity_data_structure[vertex].get(&partition_dest).unwrap_or(&0);

        for (neighbour_vertex, edge_weight) in graph.neighbors(vertex) {
            *vertex_connectivity_data_structure[neighbour_vertex].entry(partition_dest).or_insert(0) += edge_weight;
        }
    }
}

fn is_higher_placed(vertex1: usize, vertex2: usize, gain: &[Option<i64>], list_of_vertices: &FxHashSet<usize>) -> bool {
    // Checks if vertex1 is better ranked than vertex2 (used in the vertex afterburner).

    if list_of_vertices.contains(&vertex1) && !gain[vertex1].is_none()
        &&
        !gain[vertex2].is_none()
        && ((gain[vertex1].unwrap() > gain[vertex2].unwrap())
        ||
        (gain[vertex1].unwrap() == gain[vertex2].unwrap() && vertex1 < vertex2)){
        return true;
    }

    false
}

fn calculate_slot(loss: i64, max_slot_size: usize) -> usize {
    // Calculate the slot in which the vertex should be put in based on the loss value.

    if loss < 0 {
        0
    } else if loss == 0 {
        1
    } else {
        ((2 + loss.to_i64().unwrap().ilog2()) as usize).min(max_slot_size-1)
    }
}

fn get_most_connected_light_partition(vertex: usize, eligible_partitions: &FxHashSet<usize>, vertex_connectivity_data_structure: &[FxHashMap<usize, i64>]) -> Option<usize> {
    // Gets the most connection light partitions for a particular vertex.
    let mut most_strength = 0;
    let mut most_connected_partition= None;

    for (&partition, &strength) in &vertex_connectivity_data_structure[vertex] {
        if eligible_partitions.contains(&partition) {
            if strength > most_strength {
                most_strength = strength;
                most_connected_partition = Some(partition);
            }
        }
    }
    most_connected_partition
}

fn get_index_for_bucket(partition_index: usize, slot: usize, max_slots: usize) -> usize {
    // Gets the index of the bucket based on slot and partition index.

    partition_index * max_slots + slot
}

fn init_bucket(num_heavy_partitions: usize, max_slots: usize) -> Vec<Vec<usize>>{
    // Initialize the bucket where a list of vertices are stored in slots.

    let rows = num_heavy_partitions*max_slots;
    let mut bucket: Vec<Vec<usize>> = Vec::with_capacity(rows);

    for _ in 0..rows {
        bucket.push(Vec::new());
    }

    bucket
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct JetRefiner {
    // Number of partitions
    pub num_of_partitions: usize,

    // This indicates the number of times jetlp/jetrw combination should run without seeing
    // any improvement before terminating the algorithm
    pub iterations: u32,

    // A numerical factor ranging between 0.0 and 1.0 that determines the maximum allowable
    // deviation for a partition. The maximum weight of a partition with a balance factor of lambda
    // can be (1+lambda)*((totol weight of graph)/(number of partitions)).
    pub balance_factor: f64,

    // A numerical ratio ranging from 0.0 to 1.0 that determines which vertices are eligible for consideration based on
    // their gain value in the first filter. A vertice would be considered
    // if -gain(vertice) > (filter ratio)*(maximum connectivity of the vertice to any destination partition)
    pub filter_ratio: f64,

    // A numerical factor ranging from 0.0 to 1.0 that is used to determine when to reset the iteration counter.
    // If the new edge cut is less than tolerance factor times the best edge cut, then the
    // iteration counter would be reset, otherwise the iteration counter would increment
    // as it indicates the edge is becoming better at a very slow pace.
    pub tolerance_factor: f64,
}

impl<'a> crate::Partition<(Graph, &'a [i64])> for JetRefiner {
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (Graph, &'a [i64]),
    ) -> Result<Self::Metadata, Self::Error> {

        if part_ids.len() != weights.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: weights.len(),
            });
        }
        if part_ids.len() != adjacency.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: adjacency.len(),
            });
        }
        let metadata = jet_refiner(
            part_ids,
            weights,
            adjacency,
            self.num_of_partitions,
            self.iterations,
            self.balance_factor,
            self.filter_ratio,
            self.tolerance_factor,
            None
        );
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use crate::gen_weights::gen_uniform_weights;
    use crate::io::read_matrix_market_as_graph;
    use super::*;

    #[test]
    fn test_get_locked_vertices() {
        // Arrange
        let moves = vec![Move{vertex:0, partition_id:3},
                                     Move{vertex:3, partition_id:4},
                                     Move{vertex:4, partition_id:5}];
        let mut locked_vertices = [false; 5];

        // Act
        lock_vertices(&moves, &mut locked_vertices);

        // Assert
        assert!(locked_vertices[0usize]);
        assert!(locked_vertices[3usize]);
        assert!(locked_vertices[4usize]);
        assert!(!locked_vertices[2usize]);

    }
    #[test]
    fn test_init_vertex_connectivity_data_structure() {
        // Arrange
        let mut adjacency = Graph::new();
        adjacency.insert(0, 1, 2);
        adjacency.insert(0, 2, 1);
        adjacency.insert(0, 3, 4);
        adjacency.insert(1, 0, 2);
        adjacency.insert(2, 0, 1);
        adjacency.insert(3, 0, 4);

        let partition = [0, 0, 0, 1];

        // Act
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency,
            &partition);

        // Assert
        assert_eq!(vtx_conn_data_struct[0][&0], 3);
        assert_eq!(vtx_conn_data_struct[0][&1], 4);
        assert_eq!(vtx_conn_data_struct[1][&0], 2);
        assert_eq!(vtx_conn_data_struct[2][&0], 1);
        assert_eq!(vtx_conn_data_struct[3][&0], 4);
    }

    #[test]
    fn test_conn() {
        // Arrange
        let mut adjacency = Graph::new();
        adjacency.insert(0, 1, 2);
        adjacency.insert(0, 2, 1);
        adjacency.insert(0, 3, 4);
        adjacency.insert(1, 0, 2);
        adjacency.insert(2, 0, 1);
        adjacency.insert(3, 0, 4);
        let num_of_partitions = 2;

        let partition = [0, 0, 0, 1];
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency,
            &partition);

        // Act
        let conn_strength_part_0 = conn(0, 0, &vtx_conn_data_struct);
        let conn_strength_part_1 = conn(0, 1, &vtx_conn_data_struct);

        // Assert
        assert_eq!(conn_strength_part_0, 3);
        assert_eq!(conn_strength_part_1, 4);

    }

    #[test]
    fn test_gain_conn_ratio_filter() {
        // Arrange
        let mut adjacency = Graph::new();
        adjacency.insert(0, 1, 3);
        adjacency.insert(0, 2, 1);
        adjacency.insert(0, 3, 4);
        adjacency.insert(1, 0, 3);
        adjacency.insert(2, 0, 1);
        adjacency.insert(3, 0, 4);

        let partitions = [0, 0, 0, 1];
        let num_of_partitions = 2;
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency,
            &partitions);
        let gain = [Some(-1), Some(2), Some(-2), Some(-2)];
        let filter_ratio = 0.75;
        let mut locked_vertices = [false; 4];
        locked_vertices[2] = true;
        locked_vertices[3] = true;

        // Act
        let eligible_vertices_to_move = gain_conn_ratio_filter(
            &locked_vertices,
            &partitions,
            &gain,
            &vtx_conn_data_struct,
            filter_ratio);

        // Assert
        assert_eq!(eligible_vertices_to_move.len(), 2);
        assert_eq!(eligible_vertices_to_move[0], 0);
        assert_eq!(eligible_vertices_to_move[1], 1);
    }

    #[test]
    fn test_update_parts_and_vertex_connectivity(){
        // Arrange
        let mut adjacency = Graph::new();
        adjacency.insert(0, 1, 1);
        adjacency.insert(0, 2, 2);
        adjacency.insert(2, 4, 3);
        adjacency.insert(4, 5, 1);
        adjacency.insert(5, 3, 3);
        adjacency.insert(3, 1, 2);
        adjacency.insert(1, 0, 1);
        adjacency.insert(2, 0, 2);
        adjacency.insert(4, 2, 3);
        adjacency.insert(5, 4, 1);
        adjacency.insert(3, 5, 3);
        adjacency.insert(1, 3, 2);

        let num_of_partitions = 2;
        let mut partitions = [0, 0, 0, 0, 1, 1];
        let vtx_weights = [1, 1, 1, 1, 1, 1];
        let mut edge_cut = adjacency.edge_cut(&partitions);
        let mut vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency,
            &partitions);
        let moves = vec![
            Move{
                vertex: 2,
                partition_id: 1,
            },
            Move{
                vertex: 3,
                partition_id: 1,
            }
        ];
        let mut partition_weights = compute_parts_load(&partitions, 2, &vtx_weights);
        let mut dest_partition = [-1; 6];
        let mut gain = [None; 6];

        // Act
        update_parts_and_vertex_connectivity(&adjacency,
                                             &mut partitions,
                                             &mut vtx_conn_data_struct,
                                             &moves,
                                             &mut edge_cut,
                                             &mut partition_weights,
                                             &vtx_weights,
                                             &mut dest_partition,
                                             &mut gain);

        // Assert
        assert_eq!(partitions[2], 1);
        assert_eq!(partitions[3], 1);
        assert_eq!(vtx_conn_data_struct[0][&0], 1);
        assert_eq!(vtx_conn_data_struct[0][&1], 2);
        assert_eq!(vtx_conn_data_struct[1][&0], 1);
        assert_eq!(vtx_conn_data_struct[1][&1], 2);
        assert_eq!(vtx_conn_data_struct[2][&0], 2);
        assert_eq!(vtx_conn_data_struct[2][&1], 3);
        assert_eq!(vtx_conn_data_struct[3][&0], 2);
        assert_eq!(vtx_conn_data_struct[3][&1], 3);
        assert_eq!(vtx_conn_data_struct[4][&1], 4);
        assert_eq!(vtx_conn_data_struct[5][&1], 4);
        assert_eq!(edge_cut, 4);
        assert_eq!(2, partition_weights[0]);
        assert_eq!(4, partition_weights[1]);
    }

    #[test]
    fn test_is_higher_placed(){
        // Arrange
        let gain = [Some(4), Some(2), Some(2), Some(1)];
        let list_of_vertices = [0, 1, 2].into_iter().collect();

        // Act
        let result1 = is_higher_placed(0, 2, &gain, &list_of_vertices);

        // Assert
        assert_eq!(result1, true);

        // Act
        let result2 = is_higher_placed(1, 2, &gain, &list_of_vertices);

        // Assert
        assert_eq!(result2, true);

        // Act
        let result3 = is_higher_placed(3, 2, &gain, &list_of_vertices);
        // Assert
        assert_eq!(result3, false);
    }

    #[test]
    fn test_calculate_slot() {
        // Arrange and Act
        let slot1 = calculate_slot(-4, 3);
        let slot2 = calculate_slot(0, 3);
        let slot3 = calculate_slot(6, 8);
        let slot4 = calculate_slot(10, 3);

        // Assert
        assert_eq!(slot1, 0);
        assert_eq!(slot2, 1);
        assert_eq!(slot3, 4);
        assert_eq!(slot4, 2);
    }

    #[test]
    fn test_jetrw(){
        // Arrange
        let mut adjacency = Graph::new();
        adjacency.insert(0, 1, 3);
        adjacency.insert(1, 2, 3);
        adjacency.insert(2, 3, 3);
        adjacency.insert(3, 0, 3);
        adjacency.insert(1, 0, 3);
        adjacency.insert(2, 1, 3);
        adjacency.insert(3, 2, 3);
        adjacency.insert(0, 3, 3);

        let vtx_weights = [1, 4, 4, 1];
        let partitions = [0, 0, 0, 1];
        let total_weight = 10;
        let mut random_num_gen = SmallRng::from_entropy();
        let mut partition_weights = compute_parts_load(&partitions, 2, &vtx_weights);

        // Act
        let vtx_conn_data_struct =
            init_vertex_connectivity_data_structure(
                &adjacency,
                &partitions);
        let mut moves = Vec::with_capacity(partitions.len());
        jetrw(&adjacency, &partitions, &vtx_weights, total_weight, &vtx_conn_data_struct, 2, 0.1, &mut random_num_gen, &partition_weights, &mut moves);

        // Assert
        assert_eq!(moves.len(), 2);
        assert_eq!(moves[0].vertex, 0);
        assert_eq!(moves[0].partition_id, 1);
        assert_eq!(moves[1].vertex, 2);
        assert_eq!(moves[1].partition_id, 1);
    }

    #[test]
    fn test_jetlp() {
        // Arrange
        let mut adjacency = Graph::new();
        adjacency.insert(0, 1, 5);
        adjacency.insert(1, 2, 8);
        adjacency.insert(2, 3, 1);
        adjacency.insert(3, 0, 2);
        adjacency.insert(1, 0, 5);
        adjacency.insert(2, 1, 8);
        adjacency.insert(3, 2, 1);
        adjacency.insert(0, 3, 2);

        let partitions = [0, 1, 1, 0];
        let locked_vertices = [false; 4];
        let mut dest_partitions = [-2; 4];
        let mut gain = [None; 4];

        // Act
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency,
            &partitions);
        let mut moves = Vec::with_capacity(partitions.len());
        jetlp(&adjacency,
              2,
              &partitions,
              &vtx_conn_data_struct,
              &locked_vertices,
              0.3,
              &mut dest_partitions,
              &mut gain,
              &mut moves);

        // Assert
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].vertex, 0);
        assert_eq!(moves[0].partition_id, 1);
    }

    #[test]
    fn test_vt2010() {
        let graph = read_matrix_market_as_graph(Path::new("./testdata/vt2010.mtx")).unwrap();
        let weights = gen_uniform_weights(graph.len());
        let mut rng = SmallRng::seed_from_u64(5);
        let mut partition: Vec<usize> = (0..graph.len())
            .map(|_| rng.gen_range(0..2))
            .collect();

        jet_refiner(&mut partition, &weights, graph.clone(), 2, 12, 0.1, 0.75, 0.99, Some(5));
        assert_eq!(graph.edge_cut(&partition), 334753684);
    }
}