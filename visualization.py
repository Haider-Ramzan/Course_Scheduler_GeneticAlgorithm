
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import (STUDENTS, FRIEND_PAIRS,)
from fitness import calculate_fitness
from chromosome import decode_chromosome


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


STUDENT_COLORS = {
    'S1': "#FF0000", 'S2': "#00FFEE", 'S3': "#F804AB",
    'S4': "#00FF88", 'S5': "#FAC411",
}


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_convergence(all_runs_stats, save_path=None):
    ensure_dirs()
    
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'convergence.png')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    for i, stats in enumerate(all_runs_stats):
        best_list = stats['best_fitness']
        avg_list = stats['avg_fitness']
        worst_list = stats['worst_fitness']
        
        gens = range(len(best_list))
        
        if len(all_runs_stats) > 3:
            alpha_value = 0.3
        else:
            alpha_value = 0.6
            
        if i == 0:
            best_label = 'Best'
            avg_label = 'Average'
            worst_label = 'Worst'
        else:
            best_label = ''
            avg_label = ''
            worst_label = ''

        ax1.plot(gens, best_list, color='green', alpha=alpha_value, linewidth=0.8, label=best_label)
        ax1.plot(gens, avg_list, color='blue', alpha=alpha_value, linewidth=0.8, label=avg_label)
        ax1.plot(gens, worst_list, color='red', alpha=alpha_value, linewidth=0.8, label=worst_label)

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Convergence: All Runs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    
    max_gen = 0
    for s in all_runs_stats:
        current_len = len(s['best_fitness'])
        if current_len > max_gen:
            max_gen = current_len

    padded_best_runs = []
    padded_avg_runs = []

    for s in all_runs_stats:
        best_data = list(s['best_fitness'])
        avg_data = list(s['avg_fitness'])
        
        while len(best_data) < max_gen:
            best_data.append(best_data[-1])
            
        while len(avg_data) < max_gen:
            avg_data.append(avg_data[-1])
            
        padded_best_runs.append(best_data)
        padded_avg_runs.append(avg_data)

    best_m = np.array(padded_best_runs)
    avg_m = np.array(padded_avg_runs)

    gens = np.arange(max_gen)

    mean_best = np.mean(best_m, axis=0)
    std_best = np.std(best_m, axis=0)
    
    mean_avg = np.mean(avg_m, axis=0)
    std_avg = np.std(avg_m, axis=0)

    ax2.plot(gens, mean_best, color='green', lw=2, label='Best (mean)')
    ax2.fill_between(gens, mean_best - std_best, mean_best + std_best, color='green', alpha=0.2)
    
    ax2.plot(gens, mean_avg, color='blue', lw=2, label='Average (mean)')
    ax2.fill_between(gens, mean_avg - std_avg, mean_avg + std_avg, color='blue', alpha=0.2)

    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness')
    ax2.set_title('Convergence: Mean +/- Std Dev')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"plotting Convergence in {save_path}")

    

def plot_diversity(all_runs_diversity, save_path=None):
    ensure_dirs()
    
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'diversity.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, div in enumerate(all_runs_diversity):
        generation_numbers = list(range(len(div)))
        ax.plot(generation_numbers, div, alpha=0.5, lw=1, label=f'Run {i+1}')

    max_gen = 0
    for d in all_runs_diversity:
        if len(d) > max_gen:
            max_gen = len(d)

    padded_diversity_runs = []
    for d in all_runs_diversity:
        padded_run = list(d)
        shortfall = max_gen - len(d)
        last_value = d[-1]
        
        for _ in range(shortfall):
            padded_run.append(last_value)
            
        padded_diversity_runs.append(padded_run)

    div_m = np.array(padded_diversity_runs)
    mean_diversity = np.mean(div_m, axis=0)
    
    ax.plot(list(range(max_gen)), mean_diversity, 'k--', lw=2.5, label='Mean')

    threshold = 0.30
    ax.axhline(y=threshold, color='red', ls=':', lw=1.5, label=f'Threshold ({threshold:.0%})')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity')
    ax.set_title('Population Diversity Over Generations')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"plotting Diversity in {save_path}")

def visualize_schedule(chromosome, fitness_details=None, save_path=None):
    # weekly timetable of each student
    ensure_dirs()
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'best_schedule.png')

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    times = list(range(8, 18))

    decoded = decode_chromosome(chromosome)
    student_ids = sorted(decoded.keys())
    num_students = len(student_ids)

    fig, axes = plt.subplots(num_students, 1, figsize=(14, 4.5 * num_students))
    if num_students == 1:
        axes = [axes]

    col_w, row_h = 3.2, 1.0
    x0, y0 = 1.5, 0.5

    for idx, sid in enumerate(student_ids):
        ax = axes[idx]
        student_color = STUDENT_COLORS.get(sid, '#45B7D1')
        
        
        for i, day in enumerate(days):
            ax.text(x0 + i*col_w + col_w/2, y0 + len(times)*row_h + 0.3,
                    day, ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        for j, t in enumerate(times):
            y = y0 + (len(times)-1-j)*row_h
            label = f"{t} AM" if t < 12 else ("12 PM" if t == 12 else f"{t-12} PM")
            ax.text(x0-0.2, y+row_h/2, label, ha='right', va='center', fontsize=9)

       
        for i in range(len(days)+1):
            ax.plot([x0+i*col_w]*2, [y0, y0+len(times)*row_h], color='gray', lw=0.5)

        for j in range(len(times)+1):
            ax.plot([x0, x0+len(days)*col_w], [y0+j*row_h]*2, color='gray', lw=0.5)

        for c in decoded[sid]:
            cid = c['course_id']
            sec = c['section_id']
            for s in c['schedule']:
                if s['day'] not in days or s['time'] not in times:
                    continue
                di, ti = days.index(s['day']), times.index(s['time'])
                x = x0 + di * col_w
                y = y0 + (len(times)-1-ti) * row_h
                
                rect = plt.Rectangle((x+0.05, y+0.05), col_w-0.1, row_h-0.1,
                                    facecolor=student_color, edgecolor='black', lw=1.0, alpha=0.85)
                ax.add_patch(rect)
                ax.text(x+col_w/2, y+row_h/2, f"{cid}\nSec {sec}",
                        ha='center', va='center', fontsize=10, fontweight='bold')

        ax.set_title(f"Schedule for {sid} ({STUDENTS[sid]['name']})", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, x0+len(days)*col_w+0.5)
        ax.set_ylim(-0.5, y0+len(times)*row_h+1.5)
        ax.axis('off')

    if fitness_details:
        fig.suptitle(f"Weekly Timetables by Student | Overall Fitness: {fitness_details.get('fitness', 0):.4f}", 
                     fontsize=18, fontweight='bold', y=0.98)
    else:
        fig.suptitle("Weekly Timetables by Student", fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plotting Schedule -> {save_path}")


def print_schedule_statistics(chromosome):
    fitness, details = calculate_fitness(chromosome, detailed=True)
    decoded = decode_chromosome(chromosome)

    print("\n" + "=" * 70)
    print("BEST SCHEDULE STATISTICS")
    print("=" * 70)
    print(f"\nOverall Fitness: {fitness:.4f}")
    print(f"  Time Preference:     {details['time_preference']:.4f}  (30%)")
    print(f"  Gap Minimization:    {details['gap_minimization']:.4f}  (25%)")
    print(f"  Friend Satisfaction: {details['friend_satisfaction']:.4f}  (20%)")
    print(f"  Workload Balance:    {details['workload_balance']:.4f}  (15%)")
    print(f"  Lunch Break:         {details['lunch_break']:.4f}  (10%)")
    print(f"  Penalties:           {details['penalties']}")

    if details['penalty_details']:
        for p in details['penalty_details']:
            print(f"    - {p}")

    sorted_student_ids = sorted(decoded.keys())
    for sid in sorted_student_ids:
        student = STUDENTS[sid]
        courses = decoded[sid]
        
        print(f"\n--- {sid}: {student['name']} (Year {student['year']}) ---")
        print(f"  {'ID':<5} {'Course':<28} {'Sec':<5} {'Diff':<8} {'Schedule'}")
        print(f"  {'-'*5} {'-'*28} {'-'*5} {'-'*8} {'-'*35}")
        
        for c in courses:
            schedule_strings = []
            for s in c['schedule']:
                day_abbr = s['day'][:3]
                time_str = f"{s['time']}:00"
                schedule_strings.append(f"{day_abbr} {time_str}")
            
            times_str = ", ".join(schedule_strings)
            
            difficulty_labels = ['Easy', 'Medium', 'Hard']
            diff = difficulty_labels[c['difficulty'] - 1]
            
            print(f"  {c['course_id']:<5} {c['course_name']:<28} {c['section_id']:<5} {diff:<8} {times_str}")
        
        total_credits = 0
        for c in courses:
            total_credits += c['credits']
        print(f"  Credits: {total_credits}")

    print(f"\n--- Friend Overlaps ---")
    for s1, s2 in FRIEND_PAIRS:
        s1_map = {}
        for cid, sid in chromosome[s1]:
            s1_map[cid] = sid
            
        s2_map = {}
        for cid, sid in chromosome[s2]:
            s2_map[cid] = sid

        shared = set(s1_map) & set(s2_map)
        
        same = []
        for c in shared:
            if s1_map[c] == s2_map[c]:
                same.append(c)
                
        if same:
            shared_courses_str = ", ".join(same)
        else:
            shared_courses_str = "none"
            
        print(f"  {s1} <-> {s2}: {len(same)}/{len(shared)} shared ({shared_courses_str})")


def plot_operator_analysis(all_runs_operator_logs, save_path=None):
    ensure_dirs()
    
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'operator_analysis.png')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cx_totals = {}
    mut_totals = {}
    
    for log in all_runs_operator_logs:
        crossover_operations = log.get('crossover_ops', {})
        for op, cnt in crossover_operations.items():
            if op in cx_totals:
                cx_totals[op] += cnt
            else:
                cx_totals[op] = cnt
                
        mutation_operations = log.get('mutation_ops', {})
        for op, cnt in mutation_operations.items():
            if op in mut_totals:
                mut_totals[op] += cnt
            else:
                mut_totals[op] = cnt

    ax1 = axes[0]
    if cx_totals:
        ops = sorted(cx_totals.keys())
        counts = []
        for o in ops:
            counts.append(cx_totals[o])
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax1.bar(ops, counts, color=colors[:len(ops)])
        
    ax1.set_title('Crossover Operator Usage')
    ax1.set_ylabel('Count')

    ax2 = axes[1]
    if mut_totals:
        ops = sorted(mut_totals.keys())
        counts = []
        for o in ops:
            counts.append(mut_totals[o])
            
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        ax2.bar(ops, counts, color=colors[:len(ops)])
        
    ax2.set_title('Mutation Operator Usage')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"plotting Operator analysis -> {save_path}")


def plot_mutation_rates(all_runs_mutation_history, save_path=None):
    ensure_dirs()
    
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'mutation_rates.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    
    if all_runs_mutation_history:
        if all_runs_mutation_history[0]:
            history = all_runs_mutation_history[0]
            first_generation = history[0]
            operators = list(first_generation.keys())
            
            for op in operators:
                generation_numbers = list(range(len(history)))
                operator_rates = []
                
                for h in history:
                    rate = h.get(op, 0)
                    operator_rates.append(rate)
                    
                ax.plot(generation_numbers, operator_rates, lw=1.5, label=op)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Mutation Rate')
    ax.set_title('Adaptive Mutation Rates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plotting Mutation rates -> {save_path}")



def save_best_schedule(chromosome, fitness_val, run_id=0):
    ensure_dirs()
    save_path = os.path.join(OUTPUT_DIR, f'best_schedule_run{run_id}.json')
    decoded = decode_chromosome(chromosome)

    output = {'fitness': fitness_val, 'run_id': run_id, 'schedule': {}}
    for sid in sorted(decoded.keys()):
        output['schedule'][sid] = {
            'student_name': STUDENTS[sid]['name'],
            'courses': [{
                'course_id': c['course_id'], 'course_name': c['course_name'],
                'section_id': c['section_id'], 'difficulty': c['difficulty'],
                'credits': c['credits'], 'schedule': c['schedule']
            } for c in decoded[sid]]
        }

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"Schedule saved in {save_path}")
