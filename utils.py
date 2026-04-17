

import json
import os

# i will get data here, and when i do, i will prefilter the courses, like remove them if pre req not satisfied

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

COURSES = {}
STUDENTS = {}
FRIEND_PAIRS = []
SECTION_SCHEDULES = {}
COURSE_DIFFICULTY = {}
STUDENT_COURSE_POOLS = {}


def init_data():
    # Load json files
    with open(os.path.join(DATA_DIR, "course_catalog.json"), 'r') as f:
        course_data = json.load(f)
    with open(os.path.join(DATA_DIR, "student_requirements.json"), 'r') as f:
        student_data = json.load(f)

    COURSES.clear()
    COURSES.update(course_data['courses'])

    STUDENTS.clear()
    STUDENTS.update(student_data['students'])

    FRIEND_PAIRS.clear()
    FRIEND_PAIRS.extend([tuple(p) for p in student_data['friend_pairs']])

    # Build lookups
    SECTION_SCHEDULES.clear()
    COURSE_DIFFICULTY.clear()
    prereq_map = {}

    for cid, course in COURSES.items():
        COURSE_DIFFICULTY[cid] = course.get('difficulty', 1)
        prereq_map[cid] = course.get('prerequisites', [])

        for sec in course['sections']:
            SECTION_SCHEDULES[(cid, sec['section_id'])] = sec['schedule']

    # Pre-filter elective pools by prerequisites
    STUDENT_COURSE_POOLS.clear()
    for sid, stu in STUDENTS.items():

        completed = set(stu.get('completed_courses', []))
        core = list(stu['required_courses'].get('core', []))
        num_elec = stu['required_courses'].get('electives', 0)
        pool = list(stu['required_courses'].get('elective_pool', []))
        available = completed | set(core)

        eligible = []

        for e in pool:
            prerequisites = prereq_map.get(e, [])
            
            all_prerequisites_met = True
            for p in prerequisites:
                if p not in available:
                    all_prerequisites_met = False
                    break
                    
            if all_prerequisites_met:
                eligible.append(e)

        if len(eligible) < num_elec:
            for e in pool:

                if e not in eligible:
                    eligible.append(e)

                if len(eligible) >= num_elec:
                    break

        STUDENT_COURSE_POOLS[sid] = {'core': core, 'elective_pool': eligible, 'num_electives': num_elec}

    print(f" {len(COURSES)} courses, {len(STUDENTS)} students, {len(FRIEND_PAIRS)} friend pairs")
    for sid, p in STUDENT_COURSE_POOLS.items():
        print(f"  {sid}: core={p['core']}, electives={p['num_electives']} from {p['elective_pool']}")



def get_section_schedule(course_id, section_id):
    return SECTION_SCHEDULES.get((course_id, section_id), [])


def has_time_conflict(sched_a, sched_b):
    for a in sched_a:
        for b in sched_b:
            if a['day'] == b['day'] and a['time'] == b['time']:
                return True
            
    return False


def is_blocked(student_id, day, time):
    for slot in STUDENTS[student_id]['time_preferences']['blocked'].get('slots', []):
        if slot['day'] == day and slot['time'] == time:
            return True
        
    return False



def section_violates_blocked(student_id, course_id, section_id):
    for slot in get_section_schedule(course_id, section_id):
        if is_blocked(student_id, slot['day'], slot['time']):
            return True
        
    return False


def get_student_courses(student_id):
    p = STUDENT_COURSE_POOLS[student_id]
    return list(p['core']), p['num_electives'], list(p['elective_pool'])


def get_preferred_times(student_id):
    return STUDENTS[student_id]['time_preferences']['preferred'].get('time_slots', [])


def get_available_sections(student_id, course_id, current_assignments):

    #return only non-conflicting sections
    available = []

    for sec in COURSES[course_id]['sections']:
        sid = sec['section_id']
        sched = sec['schedule']

        is_section_blocked = False
        for session in sched:
            if is_blocked(student_id, session['day'], session['time']):
                is_section_blocked = True
                break
        
        if is_section_blocked:
            continue

        has_schedule_conflict = False
        for assigned_course, assigned_section in current_assignments:
            assigned_schedule = get_section_schedule(assigned_course, assigned_section)
            if has_time_conflict(sched, assigned_schedule):
                has_schedule_conflict = True
                break
        
        if has_schedule_conflict:
            continue
        
        available.append(sid)

    return available


def get_all_student_ids():
    return sorted(STUDENTS.keys())


def get_course_sections(course_id):
    section_list = []
    all_sections_data = COURSES[course_id]['sections']

    for s in all_sections_data:
        section_list.append(s['section_id'])

    return section_list


def count_courses_per_day(student_schedule):
    day_courses = {}

    for cid, sid in student_schedule:
        for slot in get_section_schedule(cid, sid):
            day_courses.setdefault(slot['day'], set()).add(cid)

    daily_counts = {}
    for d, c in day_courses.items():
        number_of_courses = len(c)
        daily_counts[d] = number_of_courses

    return daily_counts


def get_daily_schedule(student_schedule):
    daily = {}
    for cid, sid in student_schedule:
        diff = COURSE_DIFFICULTY.get(cid, 1)

        for slot in get_section_schedule(cid, sid):
            daily.setdefault(slot['day'], []).append((slot['time'], cid, sid, diff))

    for d in daily:
        daily[d].sort()

    return daily


def schedule_to_hashable(chromosome):
    final_list = []
    sorted_students = sorted(chromosome.keys())

    for sid in sorted_students:
        
        courses = chromosome[sid]
        sorted_courses = sorted(courses)
        courses_tuple = tuple(sorted_courses)
        student_package = (sid, courses_tuple)
        final_list.append(student_package)
        
    return tuple(final_list)


def deep_copy_chromosome(chromosome):
    new_chromosome = {}
    for sid, courses in chromosome.items():
        copied_courses = list(courses)
        new_chromosome[sid] = copied_courses

    return new_chromosome
