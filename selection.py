"""
selection.py - Selection Mechanisms for GA Course Scheduling
============================================================
Tournament Selection + Roulette Wheel + Elitism.
"""

# --- Selection Parameters ---
TOURNAMENT_RATE = 0.70
TOURNAMENT_SIZE = 5
ELITE_RATE = 0.10


def tournament_selection(evaluated_pop, tournament_size, rng):
    tournament = rng.sample(evaluated_pop, min(tournament_size, len(evaluated_pop)))
    return max(tournament, key=lambda x: x[0])[1]


def roulette_wheel_selection(evaluated_pop, rng):
    if not evaluated_pop:
        return None

    fitness_scores = []
    for score, chromosome in evaluated_pop:
        fitness_scores.append(score)

    worst_score = min(fitness_scores)
    
    shifted_scores = []
    for f in fitness_scores:
        positive_score = f - worst_score + 0.000001
        shifted_scores.append(positive_score)

    total_sum_of_scores = sum(shifted_scores)

    if total_sum_of_scores == 0:
        random_index = rng.randint(0, len(evaluated_pop) - 1)
        return evaluated_pop[random_index][1]

    spin_value = rng.random()
    current_cumulative_probability = 0.0
    
    for i in range(len(shifted_scores)):
        individual_score = shifted_scores[i]
        probability_slice = individual_score / total_sum_of_scores
        
        current_cumulative_probability += probability_slice
        
        if spin_value <= current_cumulative_probability:
            return evaluated_pop[i][1]
            
    return evaluated_pop[-1][1]


def apply_elitism(evaluated_pop, elite_rate):
    population_size = len(evaluated_pop)
    number_of_elites = int(population_size * elite_rate)
    
    if number_of_elites < 1:
        n_elite = 1
    else:
        n_elite = number_of_elites

    elites_only = []
    
    top_performers = evaluated_pop[:n_elite]
    
    for score, chromosome in top_performers:
        elites_only.append(chromosome)
        
    return elites_only


def select_parents(evaluated_pop, num_parents, rng):
    n_tournament = int(num_parents * TOURNAMENT_RATE)
    n_roulette = num_parents - n_tournament

    parents = []
    for _ in range(n_tournament):
        parents.append(tournament_selection(evaluated_pop, TOURNAMENT_SIZE, rng))
    for _ in range(n_roulette):
        parents.append(roulette_wheel_selection(evaluated_pop, rng))

    rng.shuffle(parents)
    return parents
