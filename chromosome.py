

#i have use dthe encoding :
# student : (subject, section), (subject, section)....

from utils import (
    COURSES, STUDENTS, FRIEND_PAIRS,
    get_student_courses, get_section_schedule, get_available_sections,
    get_course_sections, has_time_conflict, section_violates_blocked,
    get_preferred_times, get_all_student_ids, count_courses_per_day
)

RANDOM_VALID_RATIO = 0.40
GREEDY_TIME_RATIO = 0.40
GREEDY_FRIEND_RATIO = 0.20


def decode_chromosome(chromosome):
    decoded = {} # this will be a full dictionary of students id as keys and details as vlaues
    for student_id, assignments in chromosome.items():
        decoded[student_id] = []
        for cid, sid in assignments:
            info = COURSES[cid]
            decoded[student_id].append({
                'course_id': cid, 'section_id': sid,
                'course_name': info['name'],
                'difficulty': info.get('difficulty', 1),
                'credits': info.get('credits', 3),
                'schedule': get_section_schedule(cid, sid)
            })
            
    return decoded


def validate_chromosome(chromosome):
    ## cheking hard constraints 
    # i am not checkiing pre-req here because i filtered them out before the program even starts

    violations = []
    for student_id, assignments in chromosome.items():
        # Credits
        credits = sum(COURSES[c]['credits'] for c, _ in assignments)
        if credits != 15:
            violations.append(f"{student_id}: {credits} credits")

        # Time conflicts
        for i in range(len(assignments)):
            si = get_section_schedule(assignments[i][0], assignments[i][1])
            for j in range(i + 1, len(assignments)):
                sj = get_section_schedule(assignments[j][0], assignments[j][1])
                if has_time_conflict(si, sj):
                    violations.append(f"{student_id}: {assignments[i][0]} & {assignments[j][0]} conflict")
                    
        # Max per day
        for day, cnt in count_courses_per_day(assignments).items():
            if cnt > STUDENTS[student_id].get('max_courses_per_day', 3):
                violations.append(f"{student_id}: {cnt} courses on {day}")

        # Blocked
        for cid, sid in assignments:
            if section_violates_blocked(student_id, cid, sid):
                violations.append(f"{student_id}: {cid}(S{sid}) blocked")

    return len(violations) == 0, violations


def select_electives(student_id, num_electives, rng):
    temp1, temp2, pool = get_student_courses(student_id)

    if len(pool) <= num_electives:
        return list(pool)
    
    return rng.sample(pool, num_electives) # randomly pick a num_electives number of electives


def assign_random(student_id, course_list, rng):
    assignments = []
    courses = list(course_list)
    rng.shuffle(courses)

    for cid in courses:
        avail = get_available_sections(student_id, cid, assignments)

        if not avail:
            avail = get_course_sections(cid)

        if not avail:
            return None
        
        assignments.append((cid, rng.choice(avail)))
    return assignments


def assign_time_greedy(student_id, course_list, rng):
    preferred = get_preferred_times(student_id)
    assignments = []
    courses = list(course_list)
    rng.shuffle(courses)

    for cid in courses:
        avail = get_available_sections(student_id, cid, assignments)

        if not avail:
            avail = get_course_sections(cid)

        best, best_score = None, -999

        for sid in avail:

            score = 0
            schedule = get_section_schedule(cid, sid)

            for s in schedule:
                if s['time'] in preferred:
                    score += 1


            score += rng.random() * 0.1
            if score > best_score:
                best_score, best = score, sid

        if best is None:
            best = rng.choice(get_course_sections(cid))

        assignments.append((cid, best))

    return assignments


def assign_friend_greedy(student_id, course_list, existing, rng):
    assignments = []
    courses = list(course_list)
    rng.shuffle(courses)


    friend_map = {}

    for fid in STUDENTS[student_id].get('friends', []):

        if fid in existing:
            for fc, fs in existing[fid]:
                friend_map.setdefault(fc, set()).add(fs)

    for cid in courses:
        avail = get_available_sections(student_id, cid, assignments)
        if not avail:
            avail = get_course_sections(cid)

        pref = [s for s in avail if s in friend_map.get(cid, set())]
        assignments.append((cid, rng.choice(pref) if pref else rng.choice(avail)))
    return assignments


def create_random_valid(rng):
    chrom = {}
    for sid in get_all_student_ids():
        core, ne, _ = get_student_courses(sid)
        elecs = select_electives(sid, ne, rng)
        all_c = core + elecs

        for _ in range(50):
            a = assign_random(sid, all_c, rng)
            if a:
                chrom[sid] = a
                break
        else:
            chrom[sid] = [(c, rng.choice(get_course_sections(c))) for c in all_c]
    return chrom


def create_greedy_time(rng):
    chrom = {}
    for sid in get_all_student_ids():
        
        core, ne, _ = get_student_courses(sid)
        elecs = select_electives(sid, ne, rng)
        all_c = core + elecs
        a = assign_time_greedy(sid, all_c, rng)

        if not a:
            a = assign_random(sid, all_c, rng)

        if not a:
            a = [(c, rng.choice(get_course_sections(c))) for c in all_c]
        chrom[sid] = a
        
    return chrom



def create_greedy_friend(rng):
    chrom = {}
    processed = set()
    order = []
    for s1, s2 in FRIEND_PAIRS:
        if s1 not in processed: 
            order.append(s1); processed.add(s1)

        if s2 not in processed: 
            order.append(s2); processed.add(s2)

    for sid in get_all_student_ids():
        if sid not in processed: 
            order.append(sid)

    for sid in order:
        core, ne, _ = get_student_courses(sid)
        elecs = select_electives(sid, ne, rng)
        all_c = core + elecs
        a = assign_friend_greedy(sid, all_c, chrom, rng)
        if not a:
            a = assign_random(sid, all_c, rng)
        
        if not a:
            a = [(c, rng.choice(get_course_sections(c))) for c in all_c]
       
        chrom[sid] = a
    return chrom


def initialize_population(pop_size, rng):
    #40% random, 40% time-greedy, 20% friend-greedy.
    n_rand = int(pop_size * RANDOM_VALID_RATIO)
    n_time = int(pop_size * GREEDY_TIME_RATIO)
    n_friend = pop_size - n_rand - n_time

    pop = []
    for _ in range(n_rand):
        pop.append(create_random_valid(rng))

    for _ in range(n_time):
        pop.append(create_greedy_time(rng))

    for _ in range(n_friend):
        pop.append(create_greedy_friend(rng))

    # now simply cheking all valid chromosomes
    valid = 0 

    for c in pop:
        is_valid = validate_chromosome(c)[0]
       
        if is_valid == True:
            valid += 1


    print(f"{n_rand} random + {n_time} time + {n_friend} friend = {len(pop)} total, {valid} valid")
    return pop


def print_chromosome(chromosome):
    decoded = decode_chromosome(chromosome)

    for sid in sorted(decoded):
        print(f"\n  {sid} ({STUDENTS[sid]['name']}):")
        
        for c in decoded[sid]:
            times = ", ".join(f"{s['day'][:3]} {s['time']}:00" for s in c['schedule'])
            print(f"    {c['course_id']:4s} Sec {c['section_id']} [Diff:{c['difficulty']}] - {times}")
        
        print(f"    Credits: {sum(c['credits'] for c in decoded[sid])}")
