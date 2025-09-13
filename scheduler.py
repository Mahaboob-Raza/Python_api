from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import random
import copy
import numpy as np
import time
import logging
import re
from collections import defaultdict

from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Timetable GA Scheduler")

@app.get("/")
def root():
    return {"status": "ok"}


def build_slot_labels(start_time="10:00", duration=50, slots_per_day=7, lunch_after=3):
    labels = {}
    t = datetime.strptime(start_time, "%H:%M")
    for i in range(slots_per_day):
        if i == lunch_after:
            labels[i] = "Lunch Break"
        else:
            end = t + timedelta(minutes=duration)
            labels[i] = f"{t.strftime('%H:%M')}–{end.strftime('%H:%M')}"
            t = end
    return labels


# ---------------------------
# Request payload models
# ---------------------------
class SubjectSpec(BaseModel):
    name: str
    type: str  # 'theory' or 'lab'
    hours_per_week: Optional[int] = None
    rooms: Optional[List[str]] = None

class FacultySpec(BaseModel):
    name: str
    subjects: List[str]

class GenConfig(BaseModel):
    population_size: int = 500
    generations: int = 1000
    mutation_rate: float = 0.12
    tournament_k: int = 3
    seed: Optional[int] = None

class SchedulerRequest(BaseModel):
    sections: Dict[str, List[str]]
    subjects: List[SubjectSpec]
    faculty: List[FacultySpec]
    theory_rooms: List[str]
    lab_rooms: List[str]
    weeklyTheoryCount: int = 4
    maxContinuousHours: int = 3
    gen_config: GenConfig = Field(default_factory=GenConfig)

# ---------------------------
# Globals
# ---------------------------
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SLOTS_PER_DAY = 8

def initials(name: str) -> str:
    if not name:
        return ""
    name = str(name).strip()
    if re.search(r'\d', name) or len(name) <= 3:
        return name
    parts = [p for p in name.split() if p]
    if len(parts) == 1:
        return parts[0][:2].upper()
    return "".join([p[0].upper() for p in parts])

# ---------------------------
# Option builders
# ---------------------------
def build_options(subjects, faculty, theory_rooms, lab_rooms):
    fac_map = {f["name"]: set(f["subjects"]) for f in faculty}
    theory, labs = [], []

    def room_likely_for_subject(room_name: str, subj_name: str) -> bool:
        rn = room_name.lower()
        sn = subj_name.lower()
        return sn in rn or f"for {sn}" in rn

    for subj in subjects:
        subj_name = subj["name"]
        for fac, subs in fac_map.items():
            if subj_name not in subs:
                continue
            if subj["type"] == "theory":
                for room in theory_rooms:
                    theory.append({"type": "theory", "subject": subj_name, "faculty": fac, "room": room})
            else:
                allowed = subj.get("rooms") or [r for r in lab_rooms if room_likely_for_subject(r, subj_name)]
                if not allowed:
                    allowed = lab_rooms[:]
                for room in allowed:
                    labs.append({"type": "lab", "subject": subj_name, "faculty": fac, "room": room})
    return theory, labs

# ---------------------------
# Bitmask helpers
# ---------------------------
def slots_mask(start:int, length:int) -> int:
    return ((1 << length) - 1) << start

def single_slot_mask(s:int) -> int:
    return 1 << s

# ---------------------------
# Grid helpers
# ---------------------------
def make_empty_grid():
    return [[None for _ in range(SLOTS_PER_DAY)] for _ in range(len(DAYS))]

def place_lab_block(row, lab_opt, start):
    for s in range(start, start + 4):
        row[s] = {"type": "lab", "subject": lab_opt["subject"], "faculty": lab_opt["faculty"], "room": lab_opt["room"]}
    return row

def ensure_lunch(row, lab_start=None):
    lunch_idx = 4 if lab_start == 0 else 3
    if row[lunch_idx] is None:
        row[lunch_idx] = "LUNCH"

# ---------------------------
# Occupancy helpers
# ---------------------------
def create_occupancy():
    return {}, {}

def reserve_slots(occ_map, key, day, mask):
    if key not in occ_map:
        occ_map[key] = [0] * len(DAYS)
    occ_map[key][day] |= mask

def is_free(occ_map, key, day, mask):
    return key not in occ_map or (occ_map[key][day] & mask) == 0

# ---------------------------
# Individual generator
# ---------------------------
def generate_individual(sections, lab_subjects, th_opts, lab_opts, subject_expected_map):
    fac_busy, room_busy = create_occupancy()
    indiv = {sec: make_empty_grid() for sec in sections}

    lab_by_subject = {subj: [l for l in lab_opts if l["subject"] == subj] for subj in lab_subjects}
    per_section_counts = {sec: defaultdict(int) for sec in sections}

    # Place labs
    for sec in sections:
        for lsub in lab_subjects:
            choices = lab_by_subject.get(lsub, [])
            if not choices:
                continue
            random.shuffle(choices)
            days_shuffled = list(range(len(DAYS)))
            random.shuffle(days_shuffled)
            placed = False
            for d in days_shuffled:
                for start in (0, 4):
                    mask = slots_mask(start, 4)
                    for opt in choices:
                        if is_free(fac_busy, opt["faculty"], d, mask) and is_free(room_busy, opt["room"], d, mask):
                            reserve_slots(fac_busy, opt["faculty"], d, mask)
                            reserve_slots(room_busy, opt["room"], d, mask)
                            indiv[sec][d] = place_lab_block(indiv[sec][d], opt, start)
                            ensure_lunch(indiv[sec][d], start)
                            per_section_counts[sec][lsub] += 4
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break

    # Place theories
    th_opts = th_opts or [{"type": "theory", "subject": "GEN", "faculty": "TBD", "room": "R1"}]
    faculty_load, room_load = defaultdict(int), defaultdict(int)

    for sec in sections:
        for d, row in enumerate(indiv[sec]):
            if "LUNCH" not in row:
                ensure_lunch(row)
            for s in range(SLOTS_PER_DAY):
                if row[s] is None:
                    pool_size = min(8, len(th_opts))
                    candidates = random.sample(th_opts, pool_size) if pool_size > 1 else th_opts[:]
                    slot_mask = single_slot_mask(s)
                    best_opt, best_score = None, None
                    for th in candidates:
                        subj_name = th["subject"]
                        expected = subject_expected_map.get(subj_name)
                        if expected and per_section_counts[sec][subj_name] >= expected:
                            continue
                        if is_free(fac_busy, th["faculty"], d, slot_mask) and is_free(room_busy, th["room"], d, slot_mask):
                            score = faculty_load[th["faculty"]] + room_load[th["room"]]
                            if s > 0 and isinstance(row[s-1], dict) and row[s-1].get("subject") == subj_name:
                                score += 2
                            if best_score is None or score < best_score:
                                best_score, best_opt = score, th
                    if best_opt:
                        row[s] = best_opt.copy()
                        reserve_slots(fac_busy, best_opt["faculty"], d, slot_mask)
                        reserve_slots(room_busy, best_opt["room"], d, slot_mask)
                        faculty_load[best_opt["faculty"]] += 1
                        room_load[best_opt["room"]] += 1
                        per_section_counts[sec][best_opt["subject"]] += 1
    return indiv

# ---------------------------
# Fitness function
# ---------------------------
def evaluate(indiv, sections, subject_expected_map, lab_subjects_set, max_hours):
    score = 100000
    diag = {"faculty_conflicts": 0, "room_conflicts": 0, "theory_freq_mismatch": 0,
            "continuous_violation": 0, "lab_placement_issues": 0, "lunch_errors": 0}

    fac_map = {}
    room_map = {}
    subj_counts = {sec: {} for sec in sections}

    for sec in sections:
        grid = indiv[sec]
        for d, row in enumerate(grid):
            if not any(cell == "LUNCH" for cell in row):
                diag["lunch_errors"] += 1
                score -= 1000
            for s, cell in enumerate(row):
                if cell == "LUNCH":
                    continue
                if not isinstance(cell, dict):
                    # invalid cell slot
                    score -= 200
                    continue
                subj = cell["subject"]
                fac = cell["faculty"]
                room = cell["room"]
                fac_map.setdefault((d, s), []).append(fac)
                room_map.setdefault((d, s), []).append(room)
                subj_counts[sec][subj] = subj_counts[sec].get(subj, 0) + 1

    # conflicts
    for facs in fac_map.values():
        if len(facs) != len(set(facs)):
            diag["faculty_conflicts"] += 1
            score -= 200000
    for rooms in room_map.values():
        if len(rooms) != len(set(rooms)):
            diag["room_conflicts"] += 1
            score -= 150000

    # theory frequency (use subject_expected_map; skip lab_subjects_set)
    for sec, counts in subj_counts.items():
        for subj, cnt in counts.items():
            if subj in lab_subjects_set:
                continue
            expected = subject_expected_map.get(subj, 4)
            diff = abs(cnt - expected)
            if diff:
                diag["theory_freq_mismatch"] += diff
                score -= 200 * diff

    # continuous teaching penalty
    for sec in sections:
        for row in indiv[sec]:
            run = 0
            fac = None
            for cell in row:
                if cell == "LUNCH" or not isinstance(cell, dict):
                    fac = None
                    run = 0
                    continue
                if cell["faculty"] == fac:
                    run += 1
                else:
                    fac = cell["faculty"]
                    run = 1
                if run > max_hours:
                    diag["continuous_violation"] += 1
                    score -= 1000 * (run - max_hours)

    # lab placement checks (4 contiguous slots and start at 0 or 4)
    for sec in sections:
        for row in indiv[sec]:
            lab_slots = [i for i, c in enumerate(row) if isinstance(c, dict) and c.get("type") == "lab"]
            if lab_slots:
                mn, mx = min(lab_slots), max(lab_slots)
                if (mx - mn + 1) != 4 or mn not in (0, 4):
                    diag["lab_placement_issues"] += 1
                    score -= 5000

    if diag["faculty_conflicts"] or diag["room_conflicts"]:
        return -1_000_000, diag

    return score, diag

# ---------------------------
# GA operators
# ---------------------------
def tournament(pop, scores, k):
    best = int(np.random.randint(len(pop)))
    for cand in np.random.randint(0, len(pop), k - 1):
        if scores[cand] > scores[best]:
            best = int(cand)
    return pop[best]

def crossover(p1, p2):
    child1, child2 = {}, {}
    for sec in p1.keys():
        if random.random() < 0.5:
            child1[sec] = copy.deepcopy(p1[sec])
            child2[sec] = copy.deepcopy(p2[sec])
        else:
            child1[sec] = copy.deepcopy(p2[sec])
            child2[sec] = copy.deepcopy(p1[sec])
    return child1, child2

def mutation(individual, th_opts, lab_opts, mutation_rate=0.12):
    """
    Simple mutation:
      - With some probability, swap two theory slots in a section.
      - With a smaller chance, attempt to move a lab block to another valid day/start (if available).
    This is intentionally lightweight but introduces diversity.
    """
    for sec, grid in individual.items():
        # swap theory slots
        if random.random() < mutation_rate:
            # collect theory-slot coordinates
            theory_coords = []
            for d, row in enumerate(grid):
                for s, cell in enumerate(row):
                    if isinstance(cell, dict) and cell.get("type") == "theory":
                        theory_coords.append((d, s))
            if len(theory_coords) >= 2:
                a, b = random.sample(theory_coords, 2)
                da, sa = a; db, sb = b
                grid[da][sa], grid[db][sb] = grid[db][sb], grid[da][sa]

        # attempt to move a lab block
        if random.random() < (mutation_rate * 0.5):
            # find lab blocks in this section
            for d, row in enumerate(grid):
                # detect a lab block start
                for start in (0, 4):
                    if start + 3 < SLOTS_PER_DAY and all(isinstance(row[s], dict) and row[s].get("type") == "lab" for s in range(start, start+4)):
                        current_lab = row[start]["subject"]
                        current_fac = row[start]["faculty"]
                        current_room = row[start]["room"]
                        # try to move to another day/start where that room and faculty are free
                        for d2 in range(len(DAYS)):
                            if d2 == d:
                                continue
                            mask = slots_mask(start, 4)
                            # check occupancy by scanning row (quick check: ensure target slots are None)
                            target_row = grid[d2]
                            if any(isinstance(target_row[s], dict) or target_row[s] == "LUNCH" for s in range(start, start+4)):
                                continue
                            # move block: clear old slots and set new
                            for s in range(start, start+4):
                                grid[d][s] = None
                            # place new block with same faculty/room/subject
                            for s in range(start, start+4):
                                grid[d2][s] = {"type":"lab","subject":current_lab,"faculty":current_fac,"room":current_room}
                            # ensure lunches
                            ensure_lunch(grid[d], None)
                            ensure_lunch(grid[d2], start)
                            # done one mutation for this section
                            break
                        # break outer loop after handling one block
                        break
    return individual

# ---------------------------
# GA runner
# ---------------------------
def run_ga(sections, subjects, faculty, th_rooms, lab_rooms, weekly_count, max_hours, cfg):
    # Seeding
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
    else:
        seed_val = int(time.time_ns() ^ (id(cfg) & 0xFFFF))
        random.seed(seed_val)
        np.random.seed(seed_val & 0xFFFFFFFF)

    subject_expected_map = {s["name"]: (s.get("hours_per_week") or weekly_count) for s in subjects}
    th_opts, lab_opts = build_options(subjects, faculty, th_rooms, lab_rooms)

    section_list = [f"{b}-{s}" for b, secs in sections.items() for s in secs]
    lab_subs = [s["name"] for s in subjects if s["type"] == "lab"]
    lab_subjects_set = set(lab_subs)

    pop = [generate_individual(section_list, lab_subs, th_opts, lab_opts, subject_expected_map)
           for _ in range(cfg.population_size)]

    best, best_score = None, -1e12
    start_time = time.time()

    for gen in range(cfg.generations):
        scores = []
        for indiv in pop:
            sc, _ = evaluate(indiv, section_list, subject_expected_map, lab_subjects_set, max_hours)
            scores.append(sc)
            if sc > best_score:
                best_score, best = sc, copy.deepcopy(indiv)

        logging.info(f"Gen {gen+1}/{cfg.generations} best_score={best_score:.0f}")
        if best_score > 90000:
            break

        new_pop = []
        while len(new_pop) < cfg.population_size:
            p1 = tournament(pop, scores, cfg.tournament_k)
            p2 = tournament(pop, scores, cfg.tournament_k)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutation(copy.deepcopy(c1), th_opts, lab_opts, cfg.mutation_rate))
            if len(new_pop) < cfg.population_size:
                new_pop.append(mutation(copy.deepcopy(c2), th_opts, lab_opts, cfg.mutation_rate))
        pop = new_pop

    elapsed = time.time() - start_time
    logging.info(f"GA finished in {elapsed:.1f}s best_score={best_score:.0f}")

    return best, best_score, evaluate(best, section_list, subject_expected_map, lab_subjects_set, max_hours)[1]


# ---------------------------
# Render HTML
# ---------------------------
def render_html(solution, start_time="10:00", duration=50, slots_per_day=7, lunch_after=3):
    # build slot labels dynamically
    slot_labels = build_slot_labels(start_time, duration, slots_per_day, lunch_after)

    html_parts = []
    for sec, grid in solution.items():
        html_parts.append(
            f"<div class='timetable-container'><table class='tt-table' border='1' cellpadding='6'>"
        )
        html_parts.append(
            f"<thead><tr><th colspan='{slots_per_day+1}'>{sec} Timetable</th></tr></thead>"
        )

        # header row with slot times
        header = "<tr><th>Day</th>" + "".join([f"<th>{slot_labels[i]}</th>" for i in range(slots_per_day)]) + "</tr>"
        html_parts.append(header)

        html_parts.append("<tbody>")
        for di, day in enumerate(DAYS):
            row_html = f"<tr><td>{day}</td>"
            s = 0
            while s < slots_per_day:
                cell = grid[di][s]
                if cell == "LUNCH":
                    row_html += "<td>LUNCH</td>"
                    s += 1
                    continue
                if isinstance(cell, dict) and cell.get("type") == "lab":
                    start, span = s, 0
                    while (
                        s < slots_per_day
                        and isinstance(grid[di][s], dict)
                        and grid[di][s].get("type") == "lab"
                    ):
                        span += 1
                        s += 1
                    label = initials(cell.get("faculty", ""))
                    row_html += f"<td colspan='{span}'>{cell['subject']} ({label}) {cell['room']}</td>"
                    continue
                if isinstance(cell, dict) and cell.get("type") == "theory":
                    label = initials(cell.get("faculty", ""))
                    row_html += f"<td>{cell['subject']} ({label}) {cell['room']}</td>"
                    s += 1
                    continue
                row_html += f"<td>{str(cell)}</td>"
                s += 1
            row_html += "</tr>"
            html_parts.append(row_html)
        html_parts.append("</tbody></table></div>")
    return "\n".join(html_parts)

# ---------------------------
# API endpoint
# ---------------------------
@app.post("/generate")
def generate(req: SchedulerRequest):
    subjects = [s.dict() for s in req.subjects]
    faculty = [f.dict() for f in req.faculty]
    best_indiv, best_score, diagnostics = run_ga(
        req.sections, subjects, faculty, req.theory_rooms, req.lab_rooms,
        req.weeklyTheoryCount, req.maxContinuousHours, req.gen_config
    )
    if not best_indiv:
        raise HTTPException(status_code=500, detail="GA failed to produce a solution")
    return {"timetable_html": render_html(best_indiv), "score": best_score, "diagnostics": diagnostics}
