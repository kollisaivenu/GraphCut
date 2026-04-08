// This file contains the implementation of the Jet Refiner algorithm used in the refining phase
// # Reference
//
// Gilbert, Michael S., et al. "Jet: Multilevel graph partitioning on graphics processing units."
// SIAM Journal on Scientific Computing 46.5 (2024): B700-B724.

use itertools::Itertools;
use crate::algorithms::Error;
use crate::imbalance::{compute_imbalance_from_part_loads, compute_parts_load};
use num_traits::{ToPrimitive};
use rand::{Rng, SeedableRng};
use rand::rngs::{SmallRng, StdRng};
use rustc_hash::{FxHashMap, FxHashSet};
use crate::graph::Graph;
use crate::algorithms::vertex_connectivity_data_structure1::VertexConnectivityDataStructure1;
use crate::algorithms::vertex_connectivity_data_structure2::VertexConnectivityDataStructure2;
#[derive(Debug)]
struct Move {
    // Struct to store data about a move that can either lead to better edge cuts or
    // re-balance the weights.

    //The index of the vertex.
    vertex: usize,

    // The part ID of the partition where the vertex should move to.
    part_id: usize
}

fn jet_refiner(
    partition: &mut [usize],
    weights: &[i64],
    adjacency: Graph,
    num_of_parts: usize,
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
    // let mut vertex_connectivity_data_structure = init_vertex_connectivity_data_structure(&adjacency, partition);
    let mut vertex_connectivity_data_structure = VertexConnectivityDataStructure1::init_vertex_connectivity_structure(&adjacency, num_of_parts, partition);
    let mut locked_vertices = vec![false; adjacency.len()];

    let mut part_weights = compute_parts_load(&partition, num_of_parts, &weights);
    let mut imbalance_of_current_iter_partition = compute_imbalance_from_part_loads(num_of_parts, &part_weights);
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
                      num_of_parts,
                      balance_factor,
                      &mut random_num_gen,
                      &part_weights,
                      &mut moves);
                weak_rebalance_counter += 1;
            } else {
                jetrs(&adjacency,
                      &partition_iter,
                      weights,
                      total_weight,
                      &vertex_connectivity_data_structure,
                      num_of_parts,
                      balance_factor,
                      &part_weights,
                      &mut moves);
            }
        }

        // The moves from either jetlp or jetrw are applied on the current partition state.
        update_parts_and_vertex_connectivity(&adjacency,
                                             &mut partition_iter,
                                             &mut vertex_connectivity_data_structure,
                                             &moves,
                                             &mut curr_iter_partition_edge_cut,
                                             &mut part_weights,
                                             &weights,
                                             &mut dest_partititon,
                                             &mut gain);



        imbalance_of_current_iter_partition = compute_imbalance_from_part_loads(num_of_parts, &part_weights);
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

fn jetlp(graph: &Graph, partition: &[usize], vertex_connectivity_data_structure: &VertexConnectivityDataStructure1, locked_vertices: &[bool], filter_ratio: f64, dest_part: &mut [i64], gain: &mut [Option<i64>], moves: &mut Vec<Move>) {
    // iterate over all the vertices to find out which vertices provides the best gain (decrease in edge cut)
    for vertex in 0..graph.len() {
        // These are values if all the vertex belongs to the same part as its neighbours.
        let mut vertex_calculated_gain = None;
        let mut vertex_dest_part = -1;

        if !locked_vertices[vertex] && dest_part[vertex] == -2 {

            let vertex_part = partition[vertex];
            let mut connection_strength_dest = i64::MIN;
            let connection_strength_source = vertex_connectivity_data_structure.get_conn_strength(vertex,vertex_part);

            // Iterate over all parts find out which part is most connected to the vertex.
            //for (&part, &part_connection) in vertex_connectivity_data_structure[vertex].iter() {

            for (neighbor, _) in graph.neighbors(vertex) {
                let part = partition[neighbor];
                let mut part_connection = vertex_connectivity_data_structure.get_conn_strength(vertex, part);
                if part_connection > 0 &&  part != vertex_part{

                    if part_connection > connection_strength_dest {
                        connection_strength_dest = part_connection;
                        vertex_dest_part = part as i64;
                    }
                }
            }

            if vertex_dest_part >= 0  {
                vertex_calculated_gain = Some(connection_strength_dest - connection_strength_source);
            }

            dest_part[vertex] = vertex_dest_part;
            gain[vertex] = vertex_calculated_gain;
        }
    }

    // First filter is applied to check which of the vertices are eligible for moving from one part
    // to another. Either the gain should be positive or can be slightly negative (based on the filter ratio).
    // Slightly negative gain vertices are also considered in the hope that they could provide better global solutions
    let (first_filter_eligible_moves, is_vertex_moveable) = gain_conn_ratio_filter(locked_vertices,
                                                                         partition,
                                                                         &gain,
                                                                         vertex_connectivity_data_structure,
                                                                         filter_ratio);

    // A heuristic attempt is made to approximate the true gain that would occur since
    // two positive moves when applied simultaneously can be detrimental.
    for vertex_index in 0..first_filter_eligible_moves.len() {
        let vertex = first_filter_eligible_moves[vertex_index];
        let mut gain_for_vertex = 0;

        for (neighbor_vertex, edge_weight) in graph.neighbors(vertex){
            let mut part_source = partition[neighbor_vertex];

            if is_higher_placed(neighbor_vertex, vertex, &gain, &is_vertex_moveable) {
                part_source = dest_part[neighbor_vertex] as usize;
            }

            if part_source == dest_part[vertex] as usize {
                gain_for_vertex += edge_weight;
            } else if part_source == partition[vertex] {
                gain_for_vertex -= edge_weight;
            }
        }

        if gain_for_vertex > 0 {
            moves.push(Move {vertex: vertex, part_id: dest_part[vertex] as usize});
        }
    };
}

fn is_heavy_part(weight_of_part: f64, max_possible_weight: f64) -> bool {
    if weight_of_part > max_possible_weight {
        return true;
    }

    false
}

fn is_light_part(weight_of_part: f64, max_weight_of_destination: f64) -> bool {
    if weight_of_part <= max_weight_of_destination {
        return true;
    }

    false
}

fn jetrw(
    graph: &Graph,
    partition: &[usize],
    vertex_weights: &[i64],
    total_weight: i64,
    vertex_connectivity_data_structure: &VertexConnectivityDataStructure1,
    num_of_parts: usize,
    balance_factor: f64,
    random_num_gen: &mut SmallRng,
    part_weights: &[i64],
    moves: &mut Vec<Move>,
) {
    // Weaker but better rebalancer in terms of the change in edgecut
    let max_slots: usize = 25;
    let max_possible_weight_of_part =
        (1f64 + balance_factor) * (total_weight as f64) / (num_of_parts as f64);
    let num_of_vertices = graph.len();
    let mut heavy_parts = Vec::new();
    let mut light_parts = Vec::new();

    // Set what the max weight of the destination partition can be.
    // This is to prevent oscillations when the jetrw algorithm is rerun
    let mut max_weight_of_destination_part = max_possible_weight_of_part * 0.99;

    if max_weight_of_destination_part < max_possible_weight_of_part - 100f64 {
        max_weight_of_destination_part = max_possible_weight_of_part - 100f64;
    }

    for part_id in 0..num_of_parts {
        let weight_of_part = part_weights[part_id] as f64;

        if max_possible_weight_of_part < weight_of_part {
            heavy_parts.push(part_id);
        }

        if max_weight_of_destination_part >= weight_of_part {
            light_parts.push(part_id);
        }
    }

    // Create a vector that can be used to precompute the index for every heavy part.
    let mut heavy_part_to_index = vec![0; num_of_parts];

    for (index, &heavy_part) in heavy_parts.iter().enumerate() {
        heavy_part_to_index[heavy_part] = index;
    }

    // Find out the loss for each eligible vertex move (from an overweight part to an underweight part).
    // A positive loss indicates an increase in edge cut.
    let mut part_dest = vec![0usize; num_of_vertices];
    let mut loss = vec![0i64; num_of_vertices];

    for vertex in 0..num_of_vertices {
        let weight_of_part = part_weights[partition[vertex]];

        let limit = 1.5
            * (weight_of_part as f64 - ((total_weight as f64) / (num_of_parts as f64)));

        let mut calculated_loss = 0;
        let mut dest_part: usize = 0;

        if is_heavy_part(weight_of_part as f64, max_possible_weight_of_part)
            && ((vertex_weights[vertex] as f64) < limit)
        {
            let most_connected_light_partition = get_most_connected_light_part(
                vertex,
                &part_weights,
                max_weight_of_destination_part,
                vertex_connectivity_data_structure,
                graph,
                partition
            );


            if cfg!(test){
                let mut random_num_gen = StdRng::from_seed([42u8; 32]);
                let index = random_num_gen.gen_range(0..light_parts.len());
                dest_part= light_parts[index];
            } else {
                dest_part= most_connected_light_partition.unwrap_or(light_parts[random_num_gen.gen_range(0..light_parts.len())]);
            }



            calculated_loss = conn(
                vertex,
                partition[vertex],
                vertex_connectivity_data_structure,
            ) - conn(vertex, dest_part, vertex_connectivity_data_structure);
        }
        part_dest[vertex] = dest_part;
        loss[vertex] = calculated_loss;
    }

    // Slot the loss values into different buckets. This is to prevent sorting the loss values
    // which can be expensive.
    let mut bucket = init_bucket(heavy_parts.len(), max_slots);

    for vertex in 0..num_of_vertices {
        if is_heavy_part(
            part_weights[partition[vertex]] as f64,
            max_possible_weight_of_part,
        ) {
            let index = heavy_part_to_index[partition[vertex]];
            let slot = calculate_slot(loss[vertex], max_slots);
            bucket[get_index_for_bucket(index, slot, max_slots)].push(vertex);
        }
    }

    // For each of the heavy parts, decide the vertices that can be moved from the
    // heavy parts such that the increase in edge cut is minimized.
    for (index, &heavy_part) in heavy_parts.iter().enumerate() {
        let mut is_still_heavy_part = true;
        let mut m = 0f64;
        let m_max = part_weights[heavy_part] as f64 - max_possible_weight_of_part;

        for slot in 0..max_slots {
            for &vertex in &bucket[get_index_for_bucket(index, slot, max_slots)] {
                if m < m_max {
                    m = m + (vertex_weights[vertex] as f64);
                    moves.push(Move {
                        vertex,
                        part_id: part_dest[vertex],
                    });
                } else {
                    is_still_heavy_part = false;
                    break;
                }
            }

            if !is_still_heavy_part {
                break;
            }
        }
    }
}

fn get_most_connected_light_part(
    vertex: usize,
    part_weights: &[i64],
    max_dest_part_weight: f64,
    vertex_connectivity_data_structure: &VertexConnectivityDataStructure1,
    graph: &Graph,
    partition: &[usize],
) -> Option<usize> {
    // Gets the most connection light part for a particular vertex.
    let mut most_strength = 0;
    let mut most_connected_part = None;

    //for (&part, &strength) in &vertex_connectivity_data_structure[vertex] {
    for (neighbor, _) in graph.neighbors(vertex) {
        let part = partition[neighbor];
        let strength = vertex_connectivity_data_structure.get_conn_strength(vertex, part);
        if is_light_part(
            part_weights[part] as f64,
            max_dest_part_weight,
        ) {
            if strength > most_strength {
                most_strength = strength;
                most_connected_part = Some(part);
            }
        }
    }
    most_connected_part
}

fn jetrs(graph: &Graph, partition: &[usize], vertex_weights: &[i64], total_weight: i64, vertex_connectivity_data_structure: &VertexConnectivityDataStructure1, num_of_parts: usize, balance_factor: f64, part_weights: &[i64], moves: &mut Vec<Move>) {
    // Stronger but  worse rebalancer in terms of the change in edgecut
    let max_slots: usize = 50;
    let max_possible_weight_of_part = (1f64 + balance_factor)*(total_weight as f64)/(num_of_parts as f64);
    let num_of_vertices = graph.len();
    let mut heavy_parts = Vec::new();
    let mut light_parts = Vec::new();

    // Set what the max weight of the destination part can be.
    // This is to prevent oscillations when the jetrw algorithm is rerun
    let max_weight_of_dest_part = max_possible_weight_of_part*0.99;
    let mut weight_to_add_to_part = vec![0f64; num_of_parts];

    for part_id in 0..num_of_parts{
        let weight_of_part = part_weights[part_id] as f64;

        if max_possible_weight_of_part < weight_of_part {
            heavy_parts.push(part_id);
        }

        if max_weight_of_dest_part >= weight_of_part {
            light_parts.push(part_id);
            weight_to_add_to_part[part_id] = max_possible_weight_of_part - part_weights[part_id] as f64;
        }
    }


    let mut heavy_part_to_index = vec![0; num_of_parts];
    for (index, &heavy_part) in heavy_parts.iter().enumerate() {
        heavy_part_to_index[heavy_part] = index;
    }

    // Find out the avg loss for each eligible vertex move (from an overweight part to an underweight part).
    // A positive loss indicates an increase in edge cut.
    let loss: Vec<i64> = (0..num_of_vertices).map(|vertex| {
        let weight_of_partition = part_weights[partition[vertex]];
        let limit = 1.5*(weight_of_partition as f64 - ((total_weight as f64)/(num_of_parts as f64)));
        let mut calculated_loss = 0;

        if is_heavy_part(weight_of_partition as f64, max_possible_weight_of_part) && ((vertex_weights[vertex] as f64) < limit) {

            let (conn_strength, no_of_light_parts) = calculate_connection_strength_with_light_parts(vertex,
                                                                                                              part_weights,
                                                                                                              max_weight_of_dest_part,
                                                                                                              num_of_parts,
                                                                                                              vertex_connectivity_data_structure,
            graph, partition);

            calculated_loss = conn(vertex,
                                   partition[vertex],
                                   vertex_connectivity_data_structure) - ((conn_strength as f64)/(no_of_light_parts as f64)).floor() as i64;

        }
        calculated_loss
    }).collect();


    // Slot the loss values into different buckets. This is to prevent sorting the loss values
    // which can be expensive.
    let mut bucket = init_bucket(heavy_parts.len(), max_slots);

    for vertex in 0..num_of_vertices{

        if is_heavy_part(part_weights[partition[vertex]] as f64, max_possible_weight_of_part) {
            let index = heavy_part_to_index[partition[vertex]];
            let slot = calculate_slot(loss[vertex], max_slots);
            bucket[get_index_for_bucket(index, slot, max_slots)].push(vertex);
        }
    }

    for (index, &heavy_part) in heavy_parts.iter().enumerate() {
        let mut is_still_heavy = true;
        let weight_to_remove = part_weights[heavy_part] as f64 - max_possible_weight_of_part;

        let mut weight_currently_removed = 0f64;
        for slot in 0..max_slots {
            for &vertex in &bucket[get_index_for_bucket(index, slot, max_slots)] {
                if weight_currently_removed <= weight_to_remove {
                    let mut found_light_part = false;
                    // Move a vertex in a heavy part to a light part that it is connected to.

                    //for (&neighboring_part, _) in vertex_connectivity_data_structure[vertex].iter() {
                    for (neighbor, _) in graph.neighbors(vertex) {
                        let neighboring_part = partition[neighbor];
                        if is_light_part(part_weights[neighboring_part] as f64, max_weight_of_dest_part) && weight_to_add_to_part[neighboring_part] >= vertex_weights[vertex] as f64 {
                            weight_currently_removed = weight_currently_removed + (vertex_weights[vertex] as f64);
                            moves.push(Move{vertex, part_id: neighboring_part});
                            weight_to_add_to_part[neighboring_part] -= vertex_weights[vertex] as f64;
                            found_light_part = true;
                            break;
                        }
                    }
                    // If we don't find a connected light partition for a vertex to move to,
                    // then we choose a random light partition to move it to
                    if !found_light_part{
                        for &light_part in light_parts.iter() {
                            if weight_to_add_to_part[light_part] >= vertex_weights[vertex] as f64 {
                                moves.push(Move{vertex, part_id: light_part});
                                weight_to_add_to_part[light_part] -= vertex_weights[vertex] as f64;
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

fn calculate_connection_strength_with_light_parts(vertex: usize, part_weights: &[i64], max_weight_of_dest_part: f64, num_of_parts: usize, vertex_connectivity_data_structure: &VertexConnectivityDataStructure1, graph: &Graph, partition: &[usize]) -> (i64, usize) {
    // Gets the connection strength of the vertex with all the light parts and the number of light parts.
    let mut already_seen = vec![false; num_of_parts];
    let mut conn_strength = 0i64;
    let mut unique_parts_count = 0;
    // for (&part_id, &strength) in &vertex_connectivity_data_structure[vertex] {
    for (neighbor, _) in graph.neighbors(vertex) {
        let part_id = partition[neighbor];
        let strength = vertex_connectivity_data_structure.get_conn_strength(vertex, part_id);
        if is_light_part(part_weights[part_id] as f64, max_weight_of_dest_part) {
            conn_strength += strength;

            if !already_seen[part_id] {
                unique_parts_count += 1;
                already_seen[part_id] = true;
            }
        }
    }

    (conn_strength, unique_parts_count)
}

fn lock_vertices(moves: &[Move], locked_vertices: &mut [bool]) {
    // This function gets the list of locked vertices that shouldn't be moved in the subsequent iterations.
    locked_vertices.fill(false);

    for single_move in moves{
        locked_vertices[single_move.vertex] = true;
    }

}

fn gain_conn_ratio_filter(locked_vertices: &[bool], partition: &[usize], gain: &[Option<i64>], vertex_connectivity_data_structure: &VertexConnectivityDataStructure1, filter_ratio: f64) -> (Vec<usize>, Vec<bool>) {
    // Get a list of vertices that have a positive gain or slightly negative gain value (based on the filter ratio).

    let num_vertices = partition.len();
    let mut list_of_moveable_vertices  = Vec::new();
    let mut is_vertex_moveable = vec![false; num_vertices];

    for vertex in 0..num_vertices {
        if (!locked_vertices[vertex])
            &&
            !gain[vertex].is_none()
            &&
            (gain[vertex].unwrap() > 0 || -gain[vertex].unwrap() < (filter_ratio * (conn(vertex, partition[vertex], vertex_connectivity_data_structure) as f64)).floor() as i64){
            list_of_moveable_vertices.push(vertex);
            is_vertex_moveable[vertex] = true;
        }
    }

    (list_of_moveable_vertices, is_vertex_moveable)
}

fn conn(vertex_id: usize,
        part_id: usize,
        vertex_connectivity_data_structure: &VertexConnectivityDataStructure1) ->i64 {
    // Gets how well a vertex is connected to a part (adds all the edge weights connected to the partition).

    //*vertex_connectivity_data_structure[vertex_id].get(&part_id).unwrap_or(&0i64)
    vertex_connectivity_data_structure.get_conn_strength(vertex_id, part_id)
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
    vertex_connectivity_data_structure: &mut VertexConnectivityDataStructure1,
    moves: &[Move],
    curr_iter_partition_edge_cut: &mut i64,
    part_weights: &mut [i64],
    weights: &[i64],
    dest_part: &mut [i64],
    gain: &mut [Option<i64>]) {
    // Updates the partition and the vertex connectivity data structure using the given list of moves.
    for single_move in moves {
        let vertex = single_move.vertex;
        let part_source = partition[vertex];

        // *curr_iter_partition_edge_cut += vertex_connectivity_data_structure[vertex].get(&part_source).unwrap_or(&0);
        *curr_iter_partition_edge_cut += vertex_connectivity_data_structure.get_conn_strength(vertex, part_source);
        // Setting this to -2 means it needs to be recomputed in jetlp
        dest_part[vertex] = -2;
        gain[vertex] = None;

        for (neighbour_vertex, edge_weight) in graph.neighbors(vertex) {
            // *vertex_connectivity_data_structure[neighbour_vertex].entry(part_source).or_insert(0) -= edge_weight;
            vertex_connectivity_data_structure.reduce_conn_strength(neighbour_vertex, part_source, edge_weight);
            // Setting this to -2 means it needs to be recomputed in jetlp
            // if vertex_connectivity_data_structure[neighbour_vertex][&part_source] == 0 {
            //     vertex_connectivity_data_structure[neighbour_vertex].remove(&part_source);
            // }
            dest_part[neighbour_vertex] = -2;
            gain[neighbour_vertex] = None;
        }

        part_weights[part_source] -= weights[vertex];
        part_weights[single_move.part_id] += weights[vertex];
        partition[vertex] = single_move.part_id;
    }

    for single_move in moves {
        let vertex = single_move.vertex;
        let part_dest = single_move.part_id;

        // *curr_iter_partition_edge_cut -= vertex_connectivity_data_structure[vertex].get(&part_dest).unwrap_or(&0);
        *curr_iter_partition_edge_cut -= vertex_connectivity_data_structure.get_conn_strength(vertex, part_dest);
        for (neighbour_vertex, edge_weight) in graph.neighbors(vertex) {
            // *vertex_connectivity_data_structure[neighbour_vertex].entry(part_dest).or_insert(0) += edge_weight;
            vertex_connectivity_data_structure.increase_conn_strength(neighbour_vertex, part_dest, edge_weight);
        }
    }
}

fn is_higher_placed(vertex1: usize, vertex2: usize, gain: &[Option<i64>], list_of_vertices: &[bool]) -> bool {
    // Checks if vertex1 is better ranked than vertex2 (used in the vertex afterburner).

    if list_of_vertices[vertex1] && !gain[vertex1].is_none()
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

fn get_index_for_bucket(part_index: usize, slot: usize, max_slots: usize) -> usize {
    // Gets the index of the bucket based on slot and part index.

    part_index * max_slots + slot
}

fn init_bucket(num_heavy_parts: usize, max_slots: usize) -> Vec<Vec<usize>>{
    // Initialize the bucket where a list of vertices are stored in slots.

    let rows = num_heavy_parts*max_slots;
    let mut bucket: Vec<Vec<usize>> = Vec::with_capacity(rows);

    for _ in 0..rows {
        bucket.push(Vec::new());
    }

    bucket
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct JetRefiner {
    // Number of parts
    pub num_of_parts: usize,

    // This indicates the number of times jetlp/jetrw combination should run without seeing
    // any improvement before terminating the algorithm
    pub iterations: u32,

    // A numerical factor ranging between 0.0 and 1.0 that determines the maximum allowable
    // deviation for a part. The maximum weight of a part with a balance factor of lambda
    // can be (1+lambda)*((totol weight of graph)/(number of parts)).
    pub balance_factor: f64,

    // A numerical ratio ranging from 0.0 to 1.0 that determines which vertices are eligible for consideration based on
    // their gain value in the first filter. A vertice would be considered
    // if -gain(vertice) > (filter ratio)*(maximum connectivity of the vertice to any destination part)
    pub filter_ratio: f64,

    // A numerical factor ranging from 0.0 to 1.0 that is used to determine when to reset the iteration counter.
    // If the new edge cut is less than tolerance factor times the best edge cut, then the
    // iteration counter would be reset, otherwise the iteration counter would increment
    // as it indicates the edge is becoming better at a very slow pace.
    pub tolerance_factor: f64,
}

impl crate::Partition<(Graph, &[i64])> for JetRefiner {
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (Graph, &[i64]),
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
            self.num_of_parts,
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

    // #[test]
    // fn test_get_locked_vertices() {
    //     // Arrange
    //     let moves = vec![Move{vertex:0, part_id:3},
    //                                  Move{vertex:3, part_id:4},
    //                                  Move{vertex:4, part_id:5}];
    //     let mut locked_vertices = [false; 5];
    //
    //     // Act
    //     lock_vertices(&moves, &mut locked_vertices);
    //
    //     // Assert
    //     assert!(locked_vertices[0usize]);
    //     assert!(locked_vertices[3usize]);
    //     assert!(locked_vertices[4usize]);
    //     assert!(!locked_vertices[2usize]);
    //
    // }
    // #[test]
    // fn test_init_vertex_connectivity_data_structure() {
    //     // Arrange
    //     let mut adjacency = Graph::new();
    //     adjacency.insert(0, 1, 2);
    //     adjacency.insert(0, 2, 1);
    //     adjacency.insert(0, 3, 4);
    //     adjacency.insert(1, 0, 2);
    //     adjacency.insert(2, 0, 1);
    //     adjacency.insert(3, 0, 4);
    //
    //     let partition = [0, 0, 0, 1];
    //
    //     // Act
    //     let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
    //         &adjacency,
    //         &partition);
    //
    //     // Assert
    //     assert_eq!(vtx_conn_data_struct[0][&0], 3);
    //     assert_eq!(vtx_conn_data_struct[0][&1], 4);
    //     assert_eq!(vtx_conn_data_struct[1][&0], 2);
    //     assert_eq!(vtx_conn_data_struct[2][&0], 1);
    //     assert_eq!(vtx_conn_data_struct[3][&0], 4);
    // }
    //
    // #[test]
    // fn test_conn() {
    //     // Arrange
    //     let mut adjacency = Graph::new();
    //     adjacency.insert(0, 1, 2);
    //     adjacency.insert(0, 2, 1);
    //     adjacency.insert(0, 3, 4);
    //     adjacency.insert(1, 0, 2);
    //     adjacency.insert(2, 0, 1);
    //     adjacency.insert(3, 0, 4);
    //
    //     let partition = [0, 0, 0, 1];
    //     let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
    //         &adjacency,
    //         &partition);
    //
    //     // Act
    //     let conn_strength_part_0 = conn(0, 0, &vtx_conn_data_struct);
    //     let conn_strength_part_1 = conn(0, 1, &vtx_conn_data_struct);
    //
    //     // Assert
    //     assert_eq!(conn_strength_part_0, 3);
    //     assert_eq!(conn_strength_part_1, 4);
    //
    // }
    //
    // #[test]
    // fn test_gain_conn_ratio_filter() {
    //     // Arrange
    //     let mut adjacency = Graph::new();
    //     adjacency.insert(0, 1, 3);
    //     adjacency.insert(0, 2, 1);
    //     adjacency.insert(0, 3, 4);
    //     adjacency.insert(1, 0, 3);
    //     adjacency.insert(2, 0, 1);
    //     adjacency.insert(3, 0, 4);
    //
    //     let partition = [0, 0, 0, 1];
    //     let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
    //         &adjacency,
    //         &partition);
    //     let gain = [Some(-1), Some(2), Some(-2), Some(-2)];
    //     let filter_ratio = 0.75;
    //     let mut locked_vertices = [false; 4];
    //     locked_vertices[2] = true;
    //     locked_vertices[3] = true;
    //
    //     // Act
    //     let (eligible_vertices_to_move, is_vertex_moveable) = gain_conn_ratio_filter(
    //         &locked_vertices,
    //         &partition,
    //         &gain,
    //         &vtx_conn_data_struct,
    //         filter_ratio);
    //
    //     // Assert
    //     assert_eq!(eligible_vertices_to_move.len(), 2);
    //     assert_eq!(eligible_vertices_to_move[0], 0);
    //     assert_eq!(eligible_vertices_to_move[1], 1);
    //     assert_eq!(is_vertex_moveable.len(), 4);
    //     assert_eq!(is_vertex_moveable[0], true);
    //     assert_eq!(is_vertex_moveable[1], true);
    //     assert_eq!(is_vertex_moveable[2], false);
    //     assert_eq!(is_vertex_moveable[3], false);
    // }
    //
    // #[test]
    // fn test_update_parts_and_vertex_connectivity(){
    //     // Arrange
    //     let mut adjacency = Graph::new();
    //     adjacency.insert(0, 1, 1);
    //     adjacency.insert(0, 2, 2);
    //     adjacency.insert(2, 4, 3);
    //     adjacency.insert(4, 5, 1);
    //     adjacency.insert(5, 3, 3);
    //     adjacency.insert(3, 1, 2);
    //     adjacency.insert(1, 0, 1);
    //     adjacency.insert(2, 0, 2);
    //     adjacency.insert(4, 2, 3);
    //     adjacency.insert(5, 4, 1);
    //     adjacency.insert(3, 5, 3);
    //     adjacency.insert(1, 3, 2);
    //
    //     let mut partition = [0, 0, 0, 0, 1, 1];
    //     let vtx_weights = [1, 1, 1, 1, 1, 1];
    //     let mut edge_cut = adjacency.edge_cut(&partition);
    //     let mut vtx_conn_data_struct = init_vertex_connectivity_data_structure(
    //         &adjacency,
    //         &partition);
    //     let moves = vec![
    //         Move{
    //             vertex: 2,
    //             part_id: 1,
    //         },
    //         Move{
    //             vertex: 3,
    //             part_id: 1,
    //         }
    //     ];
    //     let mut part_weights = compute_parts_load(&partition, 2, &vtx_weights);
    //     let mut dest_part = [-1; 6];
    //     let mut gain = [None; 6];
    //
    //     // Act
    //     update_parts_and_vertex_connectivity(&adjacency,
    //                                          &mut partition,
    //                                          &mut vtx_conn_data_struct,
    //                                          &moves,
    //                                          &mut edge_cut,
    //                                          &mut part_weights,
    //                                          &vtx_weights,
    //                                          &mut dest_part,
    //                                          &mut gain);
    //
    //     // Assert
    //     assert_eq!(partition[2], 1);
    //     assert_eq!(partition[3], 1);
    //     assert_eq!(vtx_conn_data_struct[0][&0], 1);
    //     assert_eq!(vtx_conn_data_struct[0][&1], 2);
    //     assert_eq!(vtx_conn_data_struct[1][&0], 1);
    //     assert_eq!(vtx_conn_data_struct[1][&1], 2);
    //     assert_eq!(vtx_conn_data_struct[2][&0], 2);
    //     assert_eq!(vtx_conn_data_struct[2][&1], 3);
    //     assert_eq!(vtx_conn_data_struct[3][&0], 2);
    //     assert_eq!(vtx_conn_data_struct[3][&1], 3);
    //     assert_eq!(vtx_conn_data_struct[4][&1], 4);
    //     assert_eq!(vtx_conn_data_struct[5][&1], 4);
    //     assert_eq!(edge_cut, 4);
    //     assert_eq!(2, part_weights[0]);
    //     assert_eq!(4, part_weights[1]);
    // }
    //
    // #[test]
    // fn test_is_higher_placed(){
    //     // Arrange
    //     let gain = [Some(4), Some(2), Some(2), Some(1)];
    //     let list_of_vertices = [true, true, true, false];
    //
    //     // Act
    //     let result1 = is_higher_placed(0, 2, &gain, &list_of_vertices);
    //
    //     // Assert
    //     assert_eq!(result1, true);
    //
    //     // Act
    //     let result2 = is_higher_placed(1, 2, &gain, &list_of_vertices);
    //
    //     // Assert
    //     assert_eq!(result2, true);
    //
    //     // Act
    //     let result3 = is_higher_placed(3, 2, &gain, &list_of_vertices);
    //     // Assert
    //     assert_eq!(result3, false);
    // }
    //
    // #[test]
    // fn test_calculate_slot() {
    //     // Arrange and Act
    //     let slot1 = calculate_slot(-4, 3);
    //     let slot2 = calculate_slot(0, 3);
    //     let slot3 = calculate_slot(6, 8);
    //     let slot4 = calculate_slot(10, 3);
    //
    //     // Assert
    //     assert_eq!(slot1, 0);
    //     assert_eq!(slot2, 1);
    //     assert_eq!(slot3, 4);
    //     assert_eq!(slot4, 2);
    // }
    //
    // #[test]
    // fn test_jetrw(){
    //     // Arrange
    //     let mut adjacency = Graph::new();
    //     adjacency.insert(0, 1, 3);
    //     adjacency.insert(1, 2, 3);
    //     adjacency.insert(2, 3, 3);
    //     adjacency.insert(3, 0, 3);
    //     adjacency.insert(1, 0, 3);
    //     adjacency.insert(2, 1, 3);
    //     adjacency.insert(3, 2, 3);
    //     adjacency.insert(0, 3, 3);
    //
    //     let vtx_weights = [1, 4, 4, 1];
    //     let partition = [0, 0, 0, 1];
    //     let total_weight = 10;
    //     let mut random_num_gen = SmallRng::from_entropy();
    //     let part_weights = compute_parts_load(&partition, 2, &vtx_weights);
    //
    //     // Act
    //     let vtx_conn_data_struct =
    //         init_vertex_connectivity_data_structure(
    //             &adjacency,
    //             &partition);
    //     let mut moves = Vec::with_capacity(partition.len());
    //     jetrw(&adjacency, &partition, &vtx_weights, total_weight, &vtx_conn_data_struct, 2, 0.1, &mut random_num_gen, &part_weights, &mut moves);
    //
    //     // Assert
    //     assert_eq!(moves.len(), 2);
    //     assert_eq!(moves[0].vertex, 0);
    //     assert_eq!(moves[0].part_id, 1);
    //     assert_eq!(moves[1].vertex, 2);
    //     assert_eq!(moves[1].part_id, 1);
    // }
    //
    // #[test]
    // fn test_jetlp() {
    //     // Arrange
    //     let mut adjacency = Graph::new();
    //     adjacency.insert(0, 1, 5);
    //     adjacency.insert(1, 2, 8);
    //     adjacency.insert(2, 3, 1);
    //     adjacency.insert(3, 0, 2);
    //     adjacency.insert(1, 0, 5);
    //     adjacency.insert(2, 1, 8);
    //     adjacency.insert(3, 2, 1);
    //     adjacency.insert(0, 3, 2);
    //
    //     let partition = [0, 1, 1, 0];
    //     let locked_vertices = [false; 4];
    //     let mut dest_part = [-2; 4];
    //     let mut gain = [None; 4];
    //
    //     // Act
    //     let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
    //         &adjacency,
    //         &partition);
    //     let mut moves = Vec::with_capacity(partition.len());
    //     jetlp(&adjacency,
    //           &partition,
    //           &vtx_conn_data_struct,
    //           &locked_vertices,
    //           0.3,
    //           &mut dest_part,
    //           &mut gain,
    //           &mut moves);
    //
    //     // Assert
    //     assert_eq!(moves.len(), 1);
    //     assert_eq!(moves[0].vertex, 0);
    //     assert_eq!(moves[0].part_id, 1);
    // }

    #[test]
    fn test_vt2010_2parts() {
        let graph = read_matrix_market_as_graph(Path::new("./testdata/vt2010.mtx")).unwrap();
        let weights = gen_uniform_weights(graph.len());
        let mut rng = SmallRng::seed_from_u64(5);
        let parts = 2;
        let mut partition: Vec<usize> = (0..graph.len())
            .map(|_| rng.gen_range(0..parts))
            .collect();

        jet_refiner(&mut partition, &weights, graph.clone(), parts, 12, 0.1, 0.75, 0.99, Some(5));
        assert_eq!(graph.edge_cut(&partition), 334753684);
    }

    #[test]
    fn test_vt2010_32parts() {
        let graph = read_matrix_market_as_graph(Path::new("./testdata/vt2010.mtx")).unwrap();
        let weights = gen_uniform_weights(graph.len());
        let mut rng = SmallRng::seed_from_u64(5);
        let parts = 32;
        let mut partition: Vec<usize> = (0..graph.len())
            .map(|_| rng.gen_range(0..parts))
            .collect();

        jet_refiner(&mut partition, &weights, graph.clone(), parts, 12, 0.1, 0.75, 0.99, Some(5));
        assert_eq!(graph.edge_cut(&partition), 1043930868);
    }

    #[test]
    fn test_vt2010_64parts() {
        let graph = read_matrix_market_as_graph(Path::new("./testdata/vt2010.mtx")).unwrap();
        let weights = gen_uniform_weights(graph.len());
        let mut rng = SmallRng::seed_from_u64(5);
        let parts = 64;
        let mut partition: Vec<usize> = (0..graph.len())
            .map(|_| rng.gen_range(0..parts))
            .collect();

        jet_refiner(&mut partition, &weights, graph.clone(), parts, 12, 0.1, 0.75, 0.99, Some(5));
        assert_eq!(graph.edge_cut(&partition), 1099369279);
    }
}