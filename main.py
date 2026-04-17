

import random
import time
import os
import json
import numpy as np

from utils import init_data, schedule_to_hashable, deep_copy_chromosome
from chromosome import initialize_population, validate_chromosome, create_random_valid, decode_chromosome
from fitness import calculate_fitness, evaluate_population
from selection import select_parents, apply_elitism, ELITE_RATE
from operators import (
    crossover, mutate, get_adaptive_mutation_rates,
    MUT_RATE_SECTION_CHANGE, MUT_RATE_COURSE_SWAP, MUT_RATE_TIME_SHIFT, MUT_RATE_FRIEND_ALIGN
)
from visualization import (
    plot_convergence, plot_diversity, visualize_schedule,
    print_schedule_statistics, plot_operator_analysis,
    plot_mutation_rates, save_best_schedule, ensure_dirs
)

POPULATION_SIZE = 60
MAX_GENERATIONS = 300
PATIENCE = 40
IMPROVEMENT_THRESHOLD = 0.001
SIMILARITY_THRESHOLD = 0.85
SIMILARITY_TOLERANCE = 0.02
DIV_CHECK_INTERVAL = 10
DIV_LOW_THRESHOLD = 0.30
DIV_INJECTION_RATE = 0.20
CONSOLE_FREQUENCY = 10
NUM_RUNS = 10
BASE_SEED = 50 # base seed can be any number

def calculate_diversity(population):
    unique = set(schedule_to_hashable(c) for c in population)
    return len(unique) / len(population) if population else 0


def check_similarity_termination(evaluated_pop):
    if not evaluated_pop: 
        return False
    
    fitness_scores = []
    for score, chromosome in evaluated_pop:
        fitness_scores.append(score)
        
    if not fitness_scores: 
        return False
    
    median_fitness = np.median(fitness_scores)
    
    if median_fitness == 0: 
        return False
    
    similar_individuals_count = 0
    for f in fitness_scores:
        difference = abs(f - median_fitness)
        relative_difference = difference / (abs(median_fitness) + 0.0000000001)
        
        if relative_difference <= SIMILARITY_TOLERANCE:
            similar_individuals_count += 1

    total_population_count = len(fitness_scores)
    similarity_ratio = similar_individuals_count / total_population_count

    if similarity_ratio >= SIMILARITY_THRESHOLD:
        return True
    else:
        return False


def run_ga(seed, run_id=0):
    rng = random.Random(seed)
    base_mut_rates = {
        'section_change': MUT_RATE_SECTION_CHANGE,
        'course_swap': MUT_RATE_COURSE_SWAP,
        'time_shift': MUT_RATE_TIME_SHIFT,
        'friend_align': MUT_RATE_FRIEND_ALIGN
    }
    cur_mut_rates = dict(base_mut_rates)

    stats = {'best_fitness': [], 'avg_fitness': [], 'worst_fitness': [], 'diversity': [], 'mutation_rate_history': []}
    op_log = {'crossover_ops': {}, 'mutation_ops': {}}

    print(f"\n{'='*60}\nGA Run {run_id + 1} (Seed: {seed})\n{'='*60}")
    pop = initialize_population(POPULATION_SIZE, rng)
    eval_pop = evaluate_population(pop)
    best_fit = eval_pop[0][0]
    best_chrom = deep_copy_chromosome(eval_pop[0][1])

    stag = 0
    conv_gen = MAX_GENERATIONS

    for gen in range(MAX_GENERATIONS):
        n_elite = max(1, int(POPULATION_SIZE * ELITE_RATE))
        n_off = POPULATION_SIZE - n_elite

        elites = apply_elitism(eval_pop, ELITE_RATE)
        parents = select_parents(eval_pop, n_off * 2, rng)

        offspring = []
        for i in range(0, len(parents) - 1, 2):
            c1, c2 = crossover(parents[i], parents[i+1], rng, op_log)
            c1 = mutate(c1, rng, cur_mut_rates, op_log)
            c2 = mutate(c2, rng, cur_mut_rates, op_log)
            offspring.append(c1)

            if len(offspring) < n_off: 
                offspring.append(c2)

        while len(offspring) < n_off: 
            offspring.append(mutate(rng.choice(parents), rng, cur_mut_rates, op_log))

        new_pop = [deep_copy_chromosome(e) for e in elites] + offspring[:n_off]

        while len(new_pop) < POPULATION_SIZE: 
            new_pop.append(create_random_valid(rng))

        pop = new_pop[:POPULATION_SIZE]

        eval_pop = evaluate_population(pop)
        g_best = eval_pop[0][0]
        g_avg = sum(f for f, _ in eval_pop) / len(eval_pop)
        g_worst =  eval_pop[-1][0]

        stats['best_fitness'].append(g_best)
        stats['avg_fitness'].append(g_avg)
        stats['worst_fitness'].append(g_worst)
        stats['mutation_rate_history'].append(dict(cur_mut_rates))

        if g_best > best_fit:
            imp = (g_best - best_fit) / (abs(best_fit) + 1e-10)
            best_fit, best_chrom = g_best, deep_copy_chromosome(eval_pop[0][1])
            stag = 0 if imp > IMPROVEMENT_THRESHOLD else stag + 1

        else: 
            stag += 1

        div = calculate_diversity(pop)
        stats['diversity'].append(div)

        if gen % DIV_CHECK_INTERVAL == 0 and gen > 0:
            if div < DIV_LOW_THRESHOLD:
                n_inj = int(POPULATION_SIZE * DIV_INJECTION_RATE)
                print(f"  Gen {gen}: Low diversity ({div:.1%}), injecting {n_inj}")

                pop = [c for _, c in eval_pop[:POPULATION_SIZE - n_inj]] + [create_random_valid(rng) for _ in range(n_inj)]
                cur_mut_rates = get_adaptive_mutation_rates(base_mut_rates, div)
                eval_pop = evaluate_population(pop)


            else: 
                cur_mut_rates = dict(base_mut_rates)

        if gen % CONSOLE_FREQUENCY == 0 or gen == MAX_GENERATIONS - 1:
            print(f"  Gen {gen:4d}: Best={g_best:.4f}  Avg={g_avg:.4f}  Worst={g_worst:.4f}  Div={div:.1%}  Stag={stag}")

        if stag >= PATIENCE:
            print(f"  >> Converged at gen {gen} (patience={PATIENCE})")
            conv_gen = gen
            break
        
        if check_similarity_termination(eval_pop):
            print(f"  >> Terminated at gen {gen} (similarity)")
            conv_gen = gen
            break

    else: 
        print(f"  >> Reached max generations ({MAX_GENERATIONS})")

    valid, _ = validate_chromosome(best_chrom)
    print(f"  Result: fitness={best_fit:.4f}, valid={valid}")

    return {'best_chromosome': best_chrom, 'best_fitness': best_fit, 'convergence_generation': conv_gen, 'stats': stats, 'operator_log': op_log, 'seed': seed, 'run_id': run_id}


def run_experiment():
    print(f"\n{'#'*60}\nEXPERIMENT: {NUM_RUNS} runs, seed={BASE_SEED}\n{'#'*60}")
    start = time.time()
    res, stat_l, div_l, ops_l, mut_l = [], [], [], [], []

    for i in range(NUM_RUNS):
        r = run_ga(BASE_SEED + i, i)
        res.append(r)
        stat_l.append(r['stats'])
        div_l.append(r['stats']['diversity'])
        ops_l.append(r['operator_log'])
        mut_l.append(r['stats']['mutation_rate_history'])
        save_best_schedule(r['best_chromosome'], r['best_fitness'], i)

    el = time.time() - start
    fits = []
    convs = []

    for r in res:
        fits.append(r['best_fitness'])
        convs.append(r['convergence_generation'])

    best_idx = np.argmax(fits)
    best = res[best_idx]

    print(f"\n{'='*60}\nSUMMARY ({NUM_RUNS} runs, {el:.1f}s)\n{'='*60}")
    print(f"  Best/Mean/Worst:  {np.max(fits):.4f} / {np.mean(fits):.4f} / {np.min(fits):.4f}")
    print(f"  Mean Convergence: Gen {np.mean(convs):.0f} +/- {np.std(convs):.0f}")
    print(f"  Overall best: Run {best_idx+1} (seed={best['seed']})")

    print("\nGenerating plots...")
    plot_convergence(stat_l)
    plot_diversity(div_l)

    _, det = calculate_fitness(best['best_chromosome'], True)
    visualize_schedule(best['best_chromosome'], det)
    plot_operator_analysis(ops_l); plot_mutation_rates(mut_l)
    print_schedule_statistics(best['best_chromosome'])
    save_best_schedule(best['best_chromosome'], best['best_fitness'], 'best')

    ensure_dirs()
    smry = {'num_runs': NUM_RUNS, 'el': el, 'best': float(np.max(fits)), 'mean': float(np.mean(fits)),
            'runs': [{'run': r['run_id'], 'seed': r['seed'], 'fit': r['best_fitness'], 'conv': r['convergence_generation']} for r in res]}
    
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'experiment_summary.json')
    with open(p, 'w') as f: 
        json.dump(smry, f, indent=2)

    print(f"\nSaved to {p}")


if __name__ == '__main__':
    init_data()
    run_experiment()
