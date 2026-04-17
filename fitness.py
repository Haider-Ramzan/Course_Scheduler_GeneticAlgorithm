

from utils import (
    COURSES, STUDENTS, FRIEND_PAIRS, COURSE_DIFFICULTY,
    get_section_schedule, has_time_conflict, get_preferred_times,
    get_daily_schedule, count_courses_per_day, section_violates_blocked
)


W_TIME = 0.30
W_GAP = 0.25
W_FRIEND = 0.20
W_WORKLOAD = 0.15
W_LUNCH = 0.10

PENALTY_TIME_CONFLICT = -1000
PENALTY_MISSING_CREDITS = -800
PENALTY_TOO_MANY_COURSES = -500
PENALTY_BLOCKED_TIME = -1000


def calculate_penalties(chromosome):
    total_penalty_score = 0
    penalty_details_list = []

    for sid, assignments in chromosome.items():
        student_profile = STUDENTS[sid]
        
        current_credits = 0
        for course_id, section_id in assignments:
            course_data = COURSES[course_id]
            current_credits += course_data['credits']
            
        if current_credits != 15:
            total_penalty_score += PENALTY_MISSING_CREDITS
            error_message = f"{sid}: {current_credits} credits"
            penalty_details_list.append(error_message)

        number_of_assignments = len(assignments)
        for i in range(number_of_assignments):
            course_a, section_a = assignments[i]
            schedule_a = get_section_schedule(course_a, section_a)
            
            for j in range(i + 1, number_of_assignments):
                course_b, section_b = assignments[j]
                schedule_b = get_section_schedule(course_b, section_b)
                
                if has_time_conflict(schedule_a, schedule_b):
                    total_penalty_score += PENALTY_TIME_CONFLICT
                    conflict_message = f"{sid}: {course_a} & {course_b}"
                    penalty_details_list.append(conflict_message)

        daily_counts = count_courses_per_day(assignments)
        max_allowed = student_profile.get('max_courses_per_day', 3)
        
        for day, count in daily_counts.items():
            if count > max_allowed:
                total_penalty_score += PENALTY_TOO_MANY_COURSES
                load_message = f"{sid}: {count} on {day}"
                penalty_details_list.append(load_message)

        for course_id, section_id in assignments:
            if section_violates_blocked(sid, course_id, section_id):
                total_penalty_score += PENALTY_BLOCKED_TIME
                block_message = f"{sid}: {course_id}(S{section_id}) blocked"
                penalty_details_list.append(block_message)

    return total_penalty_score, penalty_details_list

def calculate_time_preference(chromosome):
    total = 0
    for sid, assignments in chromosome.items():
        preferred = set(get_preferred_times(sid))
        stu = STUDENTS[sid]
        pref_cnt, pen_cnt, meetings = 0, 0, 0
        for cid, sec in assignments:
            for slot in get_section_schedule(cid, sec):
                meetings += 1
                if slot['time'] in preferred:
                    pref_cnt += 1

                sp = stu.get('special_constraints', {})

                if sp.get('avoid_early') and slot['time'] == sp.get('penalty_time', 8):
                    pen_cnt += 1
        if meetings > 0:
            total += max(0, pref_cnt / meetings - pen_cnt / meetings * 0.5)

    return total / len(chromosome)


def calculate_gap_score(chromosome):
    total_gap_hours = 0
    max_possible_gaps = 0

    for sid, assignments in chromosome.items():
        daily_data = get_daily_schedule(assignments)
        
        is_commuter = False
        student_data = STUDENTS[sid]
        if 'special_constraints' in student_data:
            if student_data['special_constraints'].get('minimize_gaps', False):
                is_commuter = True
        
        if is_commuter:
            weight = 1.5
        else:
            weight = 1.0

        for day, classes in daily_data.items():
            if len(classes) <= 1:
                continue
            
            start_times = []
            for c in classes:
                start_times.append(c[0])
            
            first_class = min(start_times)
            last_class = max(start_times)
            
            total_span = last_class - first_class + 1
            number_of_classes = len(classes)
            
            empty_hours = total_span - number_of_classes
            
            total_gap_hours += empty_hours * weight
            max_possible_gaps += 8 * weight

    if max_possible_gaps > 0:
        score = 1.0 - (total_gap_hours / max_possible_gaps)
        return score
    else:
        return 1.0


def calculate_friend_satisfaction(chromosome):
    overlaps, possible = 0, 0

    for s1, s2 in FRIEND_PAIRS:
        if s1 not in chromosome or s2 not in chromosome:
            continue
        m1 = {c: s for c, s in chromosome[s1]}
        m2 = {c: s for c, s in chromosome[s2]}
        shared = set(m1) & set(m2)
        possible += len(shared)
        overlaps += sum(1 for c in shared if m1[c] == m2[c])

    return overlaps / possible if possible > 0 else 0.5


def calculate_workload_balance(chromosome):
    total_score = 0

    for sid, assignments in chromosome.items():
        daily_data = get_daily_schedule(assignments)
        daily_difficulty_sums = []
        
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for day in days_of_week:
            if day in daily_data:
                classes_on_this_day = daily_data[day]
                day_sum = 0
                for course_info in classes_on_this_day:
                    difficulty_value = course_info[3]
                    day_sum += difficulty_value
                daily_difficulty_sums.append(day_sum)
            else:
                daily_difficulty_sums.append(0)

        active_days_difficulty = []
        for d in daily_difficulty_sums:
            if d > 0:
                active_days_difficulty.append(d)

        if not active_days_difficulty:
            total_score += 0.5
            continue
        
        hard_day_penalty = 0
        for d in active_days_difficulty:
            if d >= 9:
                hard_day_penalty += 1.0
            elif d >= 7:
                hard_day_penalty += 0.5
        
        number_of_active_days = len(active_days_difficulty)
        sum_of_difficulties = sum(active_days_difficulty)
        mean_difficulty = sum_of_difficulties / number_of_active_days
        
        variance_sum = 0
        for d in active_days_difficulty:
            variance_sum += (d - mean_difficulty) ** 2
        
        standard_deviation = (variance_sum / number_of_active_days) ** 0.5
        
        consistency_score = 1 - min(standard_deviation / 6, 1)
        manageability_score = max(0, 1 - hard_day_penalty * 0.3)
        
        student_final_balance = (0.6 * consistency_score) + (0.4 * manageability_score)
        total_score += student_final_balance

    return total_score / len(chromosome)


def calculate_lunch_score(chromosome):
    total = 0
    for sid, assignments in chromosome.items():
        busy = set()

        for cid, sec in assignments:
            for slot in get_section_schedule(cid, sec):
                if slot['time'] == 12:
                    busy.add(slot['day'])
        total += min(5 - len(busy), 3) / 3.0

    return total / len(chromosome)


def calculate_fitness(chromosome, detailed=False):
    s_time = calculate_time_preference(chromosome)
    s_gap = calculate_gap_score(chromosome)
    s_friend = calculate_friend_satisfaction(chromosome)
    s_work = calculate_workload_balance(chromosome)
    s_lunch = calculate_lunch_score(chromosome)

    weighted = W_TIME * s_time + W_GAP * s_gap + W_FRIEND * s_friend + W_WORKLOAD * s_work + W_LUNCH * s_lunch
    penalty, pen_details = calculate_penalties(chromosome)
    fitness = weighted + penalty

    if detailed:
        return fitness, {
            'time_preference': s_time, 'gap_minimization': s_gap,
            'friend_satisfaction': s_friend, 'workload_balance': s_work,
            'lunch_break': s_lunch, 'weighted_score': weighted,
            'penalties': penalty, 'penalty_details': pen_details, 'fitness': fitness,
        }
    
    return fitness


def evaluate_population(population):
    evaluated = []

    for chromosome in population:
        score = calculate_fitness(chromosome)
        pair = (score, chromosome)
        evaluated.append(pair)
    
    def get_fitness_score(item):
        return item[0]

    evaluated.sort(key=get_fitness_score, reverse=True)
    
    return evaluated
