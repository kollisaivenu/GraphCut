// This file contains the implementation of the Jet Refiner algorithm used in the refining phase
// # Reference
//
// Gilbert, Michael S., et al. "Jet: Multilevel graph partitioning on graphics processing units."
// SIAM Journal on Scientific Computing 46.5 (2024): B700-B724.

use crate::algorithms::Error;
use crate::imbalance::imbalance;
use std::collections::HashSet;
use std::collections::HashMap;
use std::ops::{AddAssign, Neg, Sub, SubAssign};
use nalgebra::ComplexField;
use num_traits::{Bounded, ToPrimitive, Zero};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
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
    iterations: u32,
    balance_factor: f64,
    filter_ratio: f64,
    tolerance_factor: f64,
) {

    debug_assert!(!partition.is_empty());
    debug_assert_eq!(partition.len(), weights.len());
    debug_assert_eq!(partition.len(), adjacency.len());

    let mut partition_iter = partition.to_vec();
    let mut current_iteration = 0;
    let num_of_partitions = partition.iter().collect::<HashSet<_>>().len();
    let mut vertex_connectivity_data_structure = init_vertex_connectivity_data_structure(&adjacency,
                                                                                                        partition,
                                                                                                        num_of_partitions);
    let mut locked_vertices = vec![false; adjacency.len()];

    let mut imbalance_of_best_partition = imbalance(num_of_partitions, &partition, weights.par_iter().cloned());
    let mut best_partition_edge_cut = adjacency.edge_cut(&partition);
    let mut imbalance_of_current_iter_partition = imbalance(num_of_partitions, &partition_iter, weights.par_iter().cloned());


    while current_iteration < iterations {

        let moves;
        if imbalance_of_current_iter_partition < balance_factor {
            // the jetlp subroutine is used to generate a better partition
            moves = jetlp(&adjacency,
                          &partition_iter,
                          &vertex_connectivity_data_structure,
                          &locked_vertices,
                          filter_ratio);

            // Based on the suggested moves of the jetlp subroutine, the vertices to be moved are locked
            // to ensure that they don't become eligible to move in the next iteration.
            // This prevents oscillation of vertices
            lock_vertices(&moves, &mut locked_vertices);
        } else {
            // the jetrw subroutine is run to balance the weights of the partition
            // (should the partitions weights become highly imbalanced)
            moves = jetrw(&adjacency,
                          &partition_iter,
                          weights,
                          &vertex_connectivity_data_structure,
                          num_of_partitions,
                          balance_factor);
        }

        // The moves from either jetlp or jetrw are applied on the current partition state.
        update_parts_and_vertex_connectivity(&adjacency,
                                             &mut partition_iter,
                                             &mut vertex_connectivity_data_structure,
                                             moves);

        imbalance_of_current_iter_partition = imbalance(num_of_partitions, &partition_iter, weights.par_iter().cloned());
        let curr_iter_partition_edge_cut = adjacency.edge_cut(&partition_iter);

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

fn jetlp(graph: &Graph, partition: &[usize], vertex_connectivity_data_structure: &Vec<Vec<i64>>, locked_vertices: &[bool], filter_ratio: f64) -> Vec<Move> {

    // iterate over all the vertices to find out which vertices provides the best gain (decrease in edge cut)
    let (partition_dest, gain): (Vec<usize>, Vec<i64>) = (0..graph.len()).into_par_iter().map(|vertex| {
        let mut calculated_gain = 0;
        let mut dest_partition = 0;
        if !locked_vertices[vertex] {
            let mut neighbors_eligible_partitions = HashSet::new();

            for (neighbor_vertex, _edge_weight) in graph.neighbors(vertex) {
                if partition[neighbor_vertex] != partition[vertex] {
                    neighbors_eligible_partitions.insert(partition[neighbor_vertex]);
                }
            }
            let neighbors_eligible_partitions: Vec<usize> = neighbors_eligible_partitions.into_iter().collect();

            if !neighbors_eligible_partitions.is_empty() {
                dest_partition = get_most_connected_partition(
                    vertex,
                    &neighbors_eligible_partitions,
                    vertex_connectivity_data_structure,
                );

                calculated_gain = conn(
                    vertex,
                    dest_partition,
                    vertex_connectivity_data_structure,
                ) - conn(
                    vertex,
                    partition[vertex],
                    vertex_connectivity_data_structure,
                );

            }
        }
        (dest_partition, calculated_gain)
    }).unzip();

    // First filter is applied to check which of the vertices are eligible for moving from one partition
    // to another. Either the gain should be positive or can be slightly negative (based on the filter ratio).
    // Slightly negative gain vertices are also considered in the hope that they could provide better global solutions
    let first_filter_eligible_moves = gain_conn_ratio_filter(
        locked_vertices,
        partition,
        &gain,
        vertex_connectivity_data_structure,
        filter_ratio);

    // A heuristic attempt is made to approximate the true gain that would occur since
    // two positive moves when applied simultaneously can be detrimental.
    let first_filter_eligible_vertices = first_filter_eligible_moves.clone().into_iter().collect::<HashSet<_>>();
    let gain2: Vec<i64> = (0..first_filter_eligible_moves.len()).into_par_iter().map(|vertex_index|{
        let vertex = first_filter_eligible_moves[vertex_index];
        let mut gain_for_vertex = 0;

        for (neighbor_vertex, edge_weight) in graph.neighbors(vertex){
            let mut partition_source = partition[neighbor_vertex];

            if is_higher_placed(neighbor_vertex, vertex, &gain, &first_filter_eligible_vertices) {
                partition_source = partition_dest[neighbor_vertex];
            }

            if partition_source == partition_dest[vertex] {
                gain_for_vertex += edge_weight;
            } else if partition_source == partition[vertex]{
                gain_for_vertex -= edge_weight;
            }
        }
        gain_for_vertex
    }).collect();

    // From the newly calculated approximate gain values, moves that yield positive gain are returned.
    non_negative_gain_filter(&first_filter_eligible_moves, &partition_dest, &gain2)
}

fn jetrw(graph: &Graph, partitions: &[usize], vertex_weights: &[i64], vertex_connectivity_data_structure: &Vec<Vec<i64>>, num_partitions: usize, balance_factor: f64) -> Vec<Move> {
    let max_slots: usize = 25;
    let total_weight: i64 = vertex_weights.iter().cloned().sum();
    let max_weight_per_partitions = (1f64 + balance_factor)*(total_weight as f64)/(num_partitions as f64);
    let num_of_vertices = graph.len();
    let mut heavy_partitions: Vec<usize> = Vec::new();
    let mut light_partitions: Vec<usize> = Vec::new();

    // Set what the max weight of the destination partition can be.
    // This is to prevent oscillations when the jetrw algorithm is rerun
    let mut max_weight_dest = max_weight_per_partitions*0.99;

    if max_weight_dest < max_weight_per_partitions - 100f64 {
        max_weight_dest = max_weight_per_partitions - 100f64;
    }

    // Find out which the partitions are heavy (need to be downsized) and what partitions are light
    // (can act as valid destination partitions).
    for partition_id in 0..num_partitions{
        let weight_of_partition = get_weight_of_partition(partition_id, partitions, vertex_weights);

        if max_weight_per_partitions < weight_of_partition {
            heavy_partitions.push(partition_id);
        }

        if max_weight_dest >= weight_of_partition {
            light_partitions.push(partition_id);
        }
    }
    let mut partition_weights = vec![0.; num_partitions];

    for partition_id in 0..num_partitions{
        partition_weights[partition_id] = get_weight_of_partition(partition_id,
                                                                  partitions,
                                                                  vertex_weights);
    }

    // Find out the loss for each eligible vertex move (from an overweight partition to an underweight partition).
    // A positive loss indicates an increase in edge cut.
    let (partitions_dest, loss): (Vec<usize>, Vec<i64>) = (0..num_of_vertices).into_par_iter().map(|vertex| {
        let weight_of_partition = partition_weights[partitions[vertex]];

        let limit = 1.5*(weight_of_partition as f64 - ((total_weight as f64)/(num_partitions as f64)));

        let mut calculated_loss = 0;
        let mut dest_partition: usize = 0;

        if heavy_partitions.contains(&partitions[vertex]) && ((vertex_weights[vertex] as f64) < limit) {

            let adjacent_partitions = &get_adjacent_eligible_destination_partitions(
                graph,
                vertex,
                &partitions,
                &light_partitions);

            if adjacent_partitions.len() == 0{
                dest_partition = light_partitions[thread_rng().gen_range(0..light_partitions.len())];
            } else {
                dest_partition = get_most_connected_partition(vertex,
                                                              adjacent_partitions,
                                                              vertex_connectivity_data_structure);
            }
            calculated_loss = conn(vertex,
                                   partitions[vertex],
                                   vertex_connectivity_data_structure) -
                conn(vertex,
                     dest_partition,
                     vertex_connectivity_data_structure);
        }
        (dest_partition, calculated_loss)
    }).unzip();

    // Slot the loss values into different buckets. This is to prevent sorting the loss values
    // which can be expensive.
    let mut bucket = init_bucket(heavy_partitions.len(), max_slots);

    for vertex in 0..num_of_vertices{

        if heavy_partitions.contains(&partitions[vertex]) {
            let index = heavy_partitions.iter().position(|&x| x == partitions[vertex]).unwrap();
            let slot = calculate_slot(loss[vertex], max_slots);
            bucket[get_index_for_bucket(index, slot, max_slots)].push(vertex);
        }
    }

    // For each of the heavy partitions, decide the vertices that can be moved from the
    // heavy partitions such that the increase in edge cut is minimized.
    let mut moves = Vec::new();
    for (index, &heavy_partition) in heavy_partitions.iter().enumerate(){
        let mut m = 0f64;
        let m_max = (get_weight_of_partition(
            heavy_partition,
            partitions,
            vertex_weights) - max_weight_per_partitions);

        for slot in 0..max_slots {

            for &vertex in &bucket[get_index_for_bucket(index, slot, max_slots)] {
                //m = m + (vertex_weights[vertex]);

                if m < m_max {
                    m = m + (vertex_weights[vertex] as f64);
                    moves.push(Move{vertex, partition_id: partitions_dest[vertex]});
                }
            }
        }
    }

    moves
}

fn lock_vertices(moves: &Vec<Move>, locked_vertices: &mut [bool]) {
    // This function gets the list of locked vertices that shouldn't be moved in the subsequent iterations.
    locked_vertices.fill(false);

    for single_move in moves{
        locked_vertices[single_move.vertex] = true;
    }

}

fn gain_conn_ratio_filter(locked_vertices: &[bool], partitions: &[usize], gain: &[i64], vertex_connectivity_data_structure: &Vec<Vec<i64>>, filter_ratio: f64) -> Vec<usize> {
    // Get a list of vertices that have a positive gain or slightly negative gain value (based on the filter ratio).

    let num_vertices = partitions.len();
    let mut list_of_moveable_vertices  = Vec::new();

    for vertex in 0..num_vertices {
        if (!locked_vertices[vertex])
            &&
            (gain[vertex] > 0 || -gain[vertex] < (filter_ratio * (conn(vertex, partitions[vertex], vertex_connectivity_data_structure) as f64)).floor() as i64){
            list_of_moveable_vertices.push(vertex);
        }
    }

    list_of_moveable_vertices
}

fn non_negative_gain_filter(first_filter_eligible_moves: &[usize],
                            partition_dest: &[usize],
                            gain: &Vec<i64>) -> Vec<Move> {
    // Gets the list of moves that have positive gain after the first filter is applied.
    let mut list_of_moves: Vec<Move> = Vec::new();

    for vertex_index in (0..first_filter_eligible_moves.len()) {

        if gain[vertex_index] > 0 {
            let vertex = first_filter_eligible_moves[vertex_index];
            list_of_moves.push(Move{vertex: vertex, partition_id: partition_dest[vertex]});
        }
    }

    list_of_moves
}

fn conn(vertex_id: usize,
        partition_id: usize,
        vertex_connectivity_data_structure: &Vec<Vec<i64>>) ->i64 {
    // Gets how well a vertex is connected to a partition (adds all the edge weights connected to the partition).

    vertex_connectivity_data_structure[vertex_id][partition_id]
}

fn get_most_connected_partition(
    vertex_id: usize,
    partition_ids: &[usize],
    vertex_connectivity_data_structure: &Vec<Vec<i64>>) -> usize {
    // Get the most connected partition to a particular vertex.

    let mut connections = i64::MIN;
    let mut most_connected_partition = partition_ids[0];

    for &partition_id in partition_ids {

        if vertex_connectivity_data_structure[vertex_id][partition_id] > connections {
            connections = vertex_connectivity_data_structure[vertex_id][partition_id];
            most_connected_partition = partition_id;
        }
    }
    most_connected_partition
}

fn init_vertex_connectivity_data_structure(graph: &Graph,
                                           partition: &[usize],
                                           num_partitions: usize) -> Vec<Vec<i64>> {
    // Initialize the vertex connectivity data structure.

    let mut vertex_connectivity_data_structure = vec![vec![0; num_partitions]; partition.len()];

    let num_of_vertices = graph.len();

    for vertex in 0..num_of_vertices {

        let neighbours = graph.neighbors(vertex);
        for (neighbour_vertex, edge_weight) in neighbours {
            vertex_connectivity_data_structure[vertex][partition[neighbour_vertex]] += edge_weight;
        }
    }

    vertex_connectivity_data_structure
}

fn update_parts_and_vertex_connectivity(
    graph: &Graph,
    partition: &mut [usize],
    vertex_connectivity_data_structure: &mut Vec<Vec<i64>>,
    moves: Vec<Move>) {
    // Updates the partitions and the vertex connectivity data structure using the given list of moves.

    for single_move in &moves {
        let vertex = single_move.vertex;
        let partition_source = partition[vertex];

        for (neighbour_vertex, edge_weight) in graph.neighbors(vertex) {
            vertex_connectivity_data_structure[neighbour_vertex][partition_source] -= edge_weight;
        }

        partition[vertex] = single_move.partition_id;
    }

    for single_move in &moves {
        let vertex = single_move.vertex;
        let partition_dest = single_move.partition_id;

        for (neighbour_vertex, edge_weight) in graph.neighbors(vertex) {
            vertex_connectivity_data_structure[neighbour_vertex][partition_dest] += edge_weight;
        }
    }
}

fn is_higher_placed(vertex1: usize, vertex2: usize, gain: &[i64], list_of_vertices: &HashSet<usize>) -> bool {
    // Checks if vertex1 is better ranked than vertex2 (used in the vertex afterburner).

    if list_of_vertices.contains(&vertex1) && (gain[vertex1] > gain[vertex2] || (gain[vertex1] == gain[vertex2] && vertex1 < vertex2)){
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

fn get_adjacent_eligible_destination_partitions(
    graph: &Graph,
    vertex: usize,
    partitions: &[usize],
    eligible_partitions: &[usize]) -> Vec<usize> {
    // Gets the list of partitions belong to the neighbors of a particular vertex.

    let mut adjacent_eligible_partitions = Vec::new();

    for (neighbour, _) in graph.neighbors(vertex){

        if eligible_partitions.contains(&partitions[neighbour]) {
            adjacent_eligible_partitions.push(partitions[neighbour]);
        }
    }
    adjacent_eligible_partitions
}

fn get_weight_of_partition(partition_id: usize, partitions: &[usize], vertex_weights: &[i64]) -> f64 {
    // Gets the weight of a particular partition.

    let mut weight = 0f64;

    for (index, partition) in partitions.iter().enumerate() {

        if *partition == partition_id {
            weight += vertex_weights[index] as f64;
        }
    }

    weight
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
            self.iterations,
            self.balance_factor,
            self.filter_ratio,
            self.tolerance_factor,
        );
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;
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
            &partition,
            2);

        // Assert
        assert_eq!(vtx_conn_data_struct[0][0], 3);
        assert_eq!(vtx_conn_data_struct[0][1], 4);

    }

    #[test]
    fn test_get_most_connected_partition(){
        // Arrange
        let mut adjacency = Graph::new();
        adjacency.insert(0, 1, 2);
        adjacency.insert(0, 2, 1);
        adjacency.insert(0, 3, 4);
        adjacency.insert(1, 0, 2);
        adjacency.insert(2, 0, 1);
        adjacency.insert(3, 0, 4);

        let partition = [0, 0, 0, 1];
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency,
            &partition,
            2);

        // Act
        let most_connected_partition = get_most_connected_partition(
            0,
            &partition,
            &vtx_conn_data_struct);

        // Assert
        assert_eq!(most_connected_partition, 1);
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

        let partition = [0, 0, 0, 1];
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency,
            &partition,
            2);

        // Act
        let conn_strength_part_0 = conn(0, 0, &vtx_conn_data_struct);
        let conn_strength_part_1 = conn(0, 1, &vtx_conn_data_struct);

        // Assert
        assert_eq!(conn_strength_part_0, 3);
        assert_eq!(conn_strength_part_1, 4);

    }

    #[test]
    fn test_non_negative_gain_filter() {
        // Arrange
        let gain = vec![3, 2, -1];
        let eligible_vertices_to_move = [0, 1, 2];
        let partition_dest  = [1, 0, 1];

        // Act
        let moves = non_negative_gain_filter(
            &eligible_vertices_to_move,
            &partition_dest,
            &gain);

        // Assert
        assert_eq!(moves.len(), 2);
        assert_eq!(moves[0].vertex, 0);
        assert_eq!(moves[0].partition_id, 1);
        assert_eq!(moves[1].vertex, 1);
        assert_eq!(moves[1].partition_id, 0);
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
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency,
            &partitions,
            2);
        let gain = [-1, 2, -2, -2];
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

        let mut partitions = [0, 0, 0, 0, 1, 1];
        let mut vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency,
            &partitions,
            2);
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

        // Act
        update_parts_and_vertex_connectivity(&adjacency,
                                             &mut partitions,
                                             &mut vtx_conn_data_struct,
                                             moves);

        // Assert
        assert_eq!(partitions[2], 1);
        assert_eq!(partitions[3], 1);
        assert_eq!(vtx_conn_data_struct[0][0], 1);
        assert_eq!(vtx_conn_data_struct[0][1], 2);
        assert_eq!(vtx_conn_data_struct[1][0], 1);
        assert_eq!(vtx_conn_data_struct[1][1], 2);
        assert_eq!(vtx_conn_data_struct[4][1], 4);
        assert_eq!(vtx_conn_data_struct[5][1], 4);
    }

    #[test]
    fn test_is_higher_placed(){
        // Arrange
        let gain = [4, 2, 2, 1];
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
    fn test_get_weight_of_partition(){
        // Arrange
        let partitions = [1, 0, 0];
        let vertex_weights = [1, 2, 3];

        // Act
        let weight = get_weight_of_partition(0, &partitions, &vertex_weights);

        // Assert
        assert_eq!(weight, 5.0);
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
    fn test_get_adjacent_eligible_destination_partitions(){
        // Arrange
        let mut adjacency = Graph::new();
        adjacency.insert(0, 1, 1);
        adjacency.insert(0, 2, 2);
        adjacency.insert(0, 3, 3);
        adjacency.insert(2, 4, 3);
        adjacency.insert(1, 0, 1);
        adjacency.insert(2, 0, 2);
        adjacency.insert(3, 0, 3);
        adjacency.insert(4, 2, 3);

        let partitions = [0, 1, 3, 4, 2];
        let light_partitions = [1, 2];

        // Act
        let adjacent_eligible_partitions = get_adjacent_eligible_destination_partitions(
            &adjacency,
            0,
            &partitions,
            &light_partitions);

        // Assert
        assert_eq!(adjacent_eligible_partitions.len(), 1);
        assert_eq!(adjacent_eligible_partitions[0], 1);
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

        // Act
        let vtx_conn_data_struct =
            init_vertex_connectivity_data_structure(
                &adjacency,
                &partitions,
                2);
        let moves = jetrw(&adjacency, &partitions, &vtx_weights, &vtx_conn_data_struct, 2, 0.1);

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

        // Act
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency,
            &partitions,
             2);
        let moves = jetlp(&adjacency,
                          &partitions,
                          &vtx_conn_data_struct,
                          &locked_vertices,
                          0.3);

        // Assert
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].vertex, 0);
        assert_eq!(moves[0].partition_id, 1);
    }
}