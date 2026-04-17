
# this file has crossover and mutations

from utils import (
    STUDENTS, FRIEND_PAIRS,
    get_section_schedule, get_course_sections, get_available_sections,
    has_time_conflict, section_violates_blocked, get_all_student_ids,
    count_courses_per_day, deep_copy_chromosome
)

CX_PROBABILITY = 0.80
CX_WEIGHT_SINGLE = 0.35
CX_WEIGHT_UNIFORM = 0.35
CX_WEIGHT_COURSE = 0.30
REPAIR_MAX_ATTEMPTS = 10


MUT_RATE_SECTION_CHANGE = 0.12
MUT_RATE_COURSE_SWAP = 0.10
MUT_RATE_TIME_SHIFT = 0.08
MUT_RATE_FRIEND_ALIGN = 0.15

ADAPTIVE_DIVERSITY_THRESHOLD = 0.30
ADAPTIVE_RATE_MULTIPLIER = 1.5


def repair_chromosome(chromosome, rng):

    for sid in get_all_student_ids():
        if sid not in chromosome: 
            continue

        assignments = list(chromosome[sid])

        for _ in range(REPAIR_MAX_ATTEMPTS):
            needs_repair = False
            # Fix time conflicts
            for i in range(len(assignments)):
                ci, si = assignments[i]
                schi = get_section_schedule(ci, si)
                
                for j in range(i + 1, len(assignments)):
                    cj, sj = assignments[j]
                    schj = get_section_schedule(cj, sj)

                    if has_time_conflict(schi, schj):
                        needs_repair = True
                        other = [a for k, a in enumerate(assignments) if k != j]
                        avail = get_available_sections(sid, cj, other)

                        if avail: 
                            assignments[j] = (cj, rng.choice(avail))

                        else:
                            other2 = [a for k, a in enumerate(assignments) if k != i]
                            avail2 = get_available_sections(sid, ci, other2)
                            if avail2: 
                                assignments[i] = (ci, rng.choice(avail2))

            # Fix blocked
            for i in range(len(assignments)):
                cid, sec = assignments[i]
                if section_violates_blocked(sid, cid, sec):
                    needs_repair = True
                    other = [a for k, a in enumerate(assignments) if k != i]
                    avail = get_available_sections(sid, cid, other)
                    if avail: assignments[i] = (cid, rng.choice(avail))

            # Fix max per day
            for day, cnt in count_courses_per_day(assignments).items():
                if cnt > STUDENTS[sid].get('max_courses_per_day', 3):
                    needs_repair = True
                    day_idx = [idx for idx, (c, s) in enumerate(assignments)
                               if any(slot['day'] == day for slot in get_section_schedule(c, s))]
                    if day_idx:
                        midx = rng.choice(day_idx)
                        ci, si = assignments[midx]
                        other = [a for k, a in enumerate(assignments) if k != midx]
                        avail = get_available_sections(sid, ci, other)
                        if avail: assignments[midx] = (ci, rng.choice(avail))

            if not needs_repair: break
        chromosome[sid] = assignments
    return chromosome


def single_point_crossover(p1, p2, rng):

    students = get_all_student_ids()
    cut = rng.randint(1, len(students) - 1)
    c1, c2 = {}, {}

    for i, sid in enumerate(students):
        c1[sid] = list(p1[sid] if i < cut else p2[sid])
        c2[sid] = list(p2[sid] if i < cut else p1[sid])

    return repair_chromosome(c1, rng), repair_chromosome(c2, rng)


def uniform_crossover(p1, p2, rng):
    c1, c2 = {}, {}
    for sid in get_all_student_ids():
        if rng.random() < 0.5:
            c1[sid], c2[sid] = list(p1[sid]), list(p2[sid])
        else:
            c1[sid], c2[sid] = list(p2[sid]), list(p1[sid])

    return repair_chromosome(c1, rng), repair_chromosome(c2, rng)


def course_based_crossover(p1, p2, rng):
    c1, c2 = {}, {}

    for sid in get_all_student_ids():
        p1m = dict(p1[sid])
        p2m = dict(p2[sid])
        courses = sorted(set(p1m) | set(p2m))
        a1, a2 = [], []

        for cid in courses:
            s1, s2 = p1m.get(cid), p2m.get(cid)

            if s1 and s2:
                if rng.random() < 0.5: a1.append((cid, s1)); a2.append((cid, s2))
                else: a1.append((cid, s2)); a2.append((cid, s1))

            elif s1: 
                a1.append((cid, s1))
                a2.append((cid, s1))

            elif s2: 
                a1.append((cid, s2))
                a2.append((cid, s2))
                
        c1[sid] = a1
        c2[sid] = a2
    return repair_chromosome(c1, rng), repair_chromosome(c2, rng)


def crossover(parent1, parent2, rng, operator_log=None):
    if rng.random() > CX_PROBABILITY:
        return deep_copy_chromosome(parent1), deep_copy_chromosome(parent2)

    roll = rng.random() * (CX_WEIGHT_SINGLE + CX_WEIGHT_UNIFORM + CX_WEIGHT_COURSE)
    if roll < CX_WEIGHT_SINGLE:
        c1, c2 = single_point_crossover(parent1, parent2, rng)
        op_name = 'single_point'
    elif roll < CX_WEIGHT_SINGLE + CX_WEIGHT_UNIFORM:
        c1, c2 = uniform_crossover(parent1, parent2, rng)
        op_name = 'uniform'
    else:
        c1, c2 = course_based_crossover(parent1, parent2, rng)
        op_name = 'course_based'

    if operator_log is not None:
        operator_log.setdefault('crossover_ops', {})[op_name] = operator_log.get('crossover_ops', {}).get(op_name, 0) + 1
    return c1, c2


def mut_section_change(chrom, rng):
    c = deep_copy_chromosome(chrom)
    
    all_student_ids = get_all_student_ids()
    sid = rng.choice(all_student_ids)

    student_assignments = c[sid]

    if not student_assignments:
        return c
        
    number_of_classes = len(student_assignments)
    random_index = rng.randint(0, number_of_classes - 1)
    
    chosen_course_id, old_section_id = student_assignments[random_index]

    all_possible_sections = get_course_sections(chosen_course_id)
    
    other_sections_only = []
    for s in all_possible_sections:
        if s != old_section_id:
            other_sections_only.append(s)

    if other_sections_only:
        other_classes = []
        for k, class_info in enumerate(student_assignments):
            if k != random_index:
                other_classes.append(class_info)
        
        available_sections = get_available_sections(sid, chosen_course_id, other_classes)
        
        conflict_free_sections = []
        for s in other_sections_only:
            if s in available_sections:
                conflict_free_sections.append(s)
        
        if conflict_free_sections:
            random_new_section = rng.choice(conflict_free_sections)
            student_assignments[random_index] = (chosen_course_id, random_new_section)
        else:
            forced_random_section = rng.choice(other_sections_only)
            student_assignments[random_index] = (chosen_course_id, forced_random_section)
            
    return c

def mut_course_swap(chrom, rng):
    c = deep_copy_chromosome(chrom)
    
    all_student_ids = get_all_student_ids()
    sid = rng.choice(all_student_ids)
    
    student_assignments = c[sid]

    if len(student_assignments) < 2: 
        return c
        
    indices = range(len(student_assignments))
    random_indices = rng.sample(indices, 2)
    i = random_indices[0]
    j = random_indices[1]

    course_i_id, section_i_old = student_assignments[i]
    course_j_id, section_j_old = student_assignments[j]

    others_for_i = []
    for k, class_info in enumerate(student_assignments):
        if k != i and k != j:
            others_for_i.append(class_info)
            
    available_for_i = get_available_sections(sid, course_i_id, others_for_i)

    if available_for_i:
        new_section_i = rng.choice(available_for_i)
        student_assignments[i] = (course_i_id, new_section_i)
        
        others_for_j = []
        for k, class_info in enumerate(student_assignments):
            if k != j:
                others_for_j.append(class_info)
                
        available_for_j = get_available_sections(sid, course_j_id, others_for_j)

        if available_for_j: 
            new_section_j = rng.choice(available_for_j)
            student_assignments[j] = (course_j_id, new_section_j)

    return c


def mut_time_shift(chrom, rng):
    c = deep_copy_chromosome(chrom)
    
    all_ids = get_all_student_ids()
    sid = rng.choice(all_ids)
    
    student_assignments = c[sid]

    if not student_assignments: 
        return c
        
    number_of_classes = len(student_assignments)
    random_index = rng.randint(0, number_of_classes - 1)
    
    chosen_course_id, old_section_id = student_assignments[random_index]

    other_classes = []
    for k, class_info in enumerate(student_assignments):
        if k != random_index:
            other_classes.append(class_info)
            
    all_possible_sections = get_available_sections(sid, chosen_course_id, other_classes)
    
    valid_new_sections = []
    for s in all_possible_sections:
        if s != old_section_id:
            valid_new_sections.append(s)
    
    if valid_new_sections: 
        random_new_section = rng.choice(valid_new_sections)
        student_assignments[random_index] = (chosen_course_id, random_new_section)
        
    return c


def mut_friend_align(chrom, rng):
    c = deep_copy_chromosome(chrom)

    if not FRIEND_PAIRS: 
        return c
    
    s1, s2 = rng.choice(FRIEND_PAIRS)

    if s1 not in c or s2 not in c: 
        return c
    
    m1 = {}
    for index, (course_id, section_id) in enumerate(c[s1]):
        m1[course_id] = (index, section_id)
        
    m2 = {}
    for index, (course_id, section_id) in enumerate(c[s2]):
        m2[course_id] = (index, section_id)

    shared_courses = set(m1.keys()) & set(m2.keys())

    if not shared_courses: 
        return c

    chosen_course = rng.choice(list(shared_courses))
    
    index1, section1 = m1[chosen_course]
    index2, section2 = m2[chosen_course]

    if section1 == section2: 
        return c

    others2 = []
    for k, x in enumerate(c[s2]):
        if k != index2:
            others2.append(x)

    available_for_s2 = get_available_sections(s2, chosen_course, others2)
    
    if section1 in available_for_s2: 
        c[s2][index2] = (chosen_course, section1)
    
    else:
        others1 = []
        for k, x in enumerate(c[s1]):
            if k != index1:
                others1.append(x)

        available_for_s1 = get_available_sections(s1, chosen_course, others1)

        if section2 in available_for_s1: 
            c[s1][index1] = (chosen_course, section2)

        else:
            all_possible_sections = get_course_sections(chosen_course)
            
            for s in all_possible_sections:
                if s == section1 or s == section2:
                    continue
                
                s1_can_take = s in get_available_sections(s1, chosen_course, others1)
                s2_can_take = s in get_available_sections(s2, chosen_course, others2)
                
                if s1_can_take and s2_can_take:
                    c[s1][index1] = (chosen_course, s)
                    c[s2][index2] = (chosen_course, s)
                    break
                    
    return c
def mutate(chrom, rng, rates=None, op_log=None):

    if rates is None:
        rates = {
            'section_change': MUT_RATE_SECTION_CHANGE,
            'course_swap': MUT_RATE_COURSE_SWAP,
            'time_shift': MUT_RATE_TIME_SHIFT,
            'friend_align': MUT_RATE_FRIEND_ALIGN
        }

    c = deep_copy_chromosome(chrom)

    ops = [
        ('section_change', rates['section_change'], mut_section_change),
        ('course_swap', rates['course_swap'], mut_course_swap),
        ('time_shift', rates['time_shift'], mut_time_shift),
        ('friend_align', rates['friend_align'], mut_friend_align)
    ]

    for name, rate, func in ops:
        if rng.random() < rate:
            c = func(c, rng)
            if op_log is not None:
                op_log.setdefault('mutation_ops', {})[name] = op_log.get('mutation_ops', {}).get(name, 0) + 1
    
    return c


def get_adaptive_mutation_rates(base_rates, diversity):
    if diversity < ADAPTIVE_DIVERSITY_THRESHOLD:
        print(f" Diversity={diversity:.1%} < {ADAPTIVE_DIVERSITY_THRESHOLD:.0%}, rates x{ADAPTIVE_RATE_MULTIPLIER}")

        return {op: min(r * ADAPTIVE_RATE_MULTIPLIER, 0.95) for op, r in base_rates.items()}
    
    return dict(base_rates)
