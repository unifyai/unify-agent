"""Regenerate ``claims.tsv`` (guard J1) from a render (``python -m tests.system.render OUT.json`` on a keyless tree) and
the recorded first requests: ``python tests/system/tools/gen_claims.py RENDER.json tests/system/render.py
tests/system/claims.tsv goldens/first_requests-*.recorded.json``. The rules below are the reviewer's classification
(checked, false, non-factual, untestable), sentence by sentence; a sentence no rule names comes out ``unchecked``.
Review every changed row before committing it. Stdlib only."""
import collections
import importlib.util
import json
import re
import sys

spec = importlib.util.spec_from_file_location("render", sys.argv[2])
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
d = json.load(open(sys.argv[1]))
for extra in sys.argv[4:]:  # the recorded first requests (dynamic texts)
    d.update(json.load(open(extra)))
sc = R.scoped(d)
T = "tests/system/"
S01 = T+"test_s01_capture_reuse.py::test_a_capture_is_served_at_once_reused_with_no_arguments_and_its_use_is_recorded"
S04 = T+"test_s04_failed_use.py::test_a_failed_use_is_recorded_not_reoffered_opens_an_entry_and_reaches_the_card"
S04R = T+"test_s04_failed_use.py::test_retire_is_reachable_and_withdraws_without_deleting"
S05 = T+"test_s05_paths_open.py::test_every_path_and_handle_shown_opens_and_refusals_name_a_working_call"
S06 = T+"test_s06_memory_off.py::test_memory_off_has_no_memory_text_no_memory_agents_and_the_same_tools_but_memory"
S07 = T+"test_s07_start_block_and_ask.py::test_the_block_and_the_ask_answer_are_one_primary_and_one_fallback_bounded"
S09 = T+"test_s09_gate_raw_candidates.py"
S12 = T+"test_s12_texts_sent_to_models.py::test_no_text_sent_to_any_model_steers_storage_or_names_a_world"
J2 = T+"test_j2_goldens.py"
J3 = T+"test_j3_constants.py"
T13 = T+"test_s13_role_claims.py::"
S13A = T13+"test_a_writers_helper_has_the_box_and_roots_its_texts_state"
S13B = T13+"test_the_writers_capture_tools_do_what_their_texts_state"
S13C = T13+"test_each_roles_refusals_name_only_roots_it_can_open"
S13D = T13+"test_the_librarians_notes_reach_a_later_librarian_and_search_lists_every_item"
S01B = T+"test_s01_capture_reuse.py::test_a_capture_whose_episode_has_no_verdict_never_claims_acceptance"
S11C = T+"test_s11_bounds_and_kinds.py::test_a_card_is_bounded_and_shows_code_only_on_request"
S11F = T+"test_s11_bounds_and_kinds.py::test_the_librarian_gets_a_fresh_context_for_each_question"
S11K = T+"test_s11_bounds_and_kinds.py::test_one_listing_two_labelled_kinds_and_a_note_served_as_its_rule"
S02 = T+"test_s02_group_and_writer_pass.py::test_a_related_capture_forms_a_group_the_pass_lands_an_item_linked_both_ways"
S13E = T13+"test_the_writers_inputs_hold_whole_episodes_and_their_blobs"
FIXT = "tests/memory/test_case_fixtures.py::test_an_edited_or_invented_case_does_not_re_derive"
J3N = T+"test_j3_constants.py::test_each_stated_number_is_its_constant"
S03 = T+"test_s03_lone_capture.py::test_a_lone_capture_waits_stays_servable_and_is_counted_never_grouped"
S00B = T+"test_s00_no_real_model.py::test_every_model_call_of_a_memory_run_is_scripted_and_boxes_have_no_network"
S02R = T+"test_s02b_relations_and_history_bound.py::test_the_pass_is_shown_only_the_items_its_captures_used_or_are_linked_to"
ROUND4 = [
 (r"^When your code will use the answer, pass `schema=`", "checked", "tests/memory/test_ask_structured.py", "a unit-level checker (the builder's): no system test asks with a schema"),
 (r"^At the start of a request, `memory\.ask` names", "checked", S07, ""),
 (r"^Your box: /workspace and a private /tmp are writable; there is no network", "checked", S00B, ""),
 (r"^Your working directory is `/workspace`; it persists across tasks", "checked", S03, ""),
 (r"^Your memory library holds \d+ functions?, \d+ notes? and \d+ recorded procedures?|^Best fit: |^call: memory\.get\(", "checked", S07, ""),
 (r"^On the record you are `librarian-<n>`", "nonfactual", "", "an identity line: the name is generated each run"),
 (r"^Pass <id16>: a group of \d+ related captures", "checked", S02, ""),
 (r"group\.json says why the harness put them together", "checked", S02R, ""),
 (r"^What each entry owes a read is in /inputs/attention/index\.json", "checked", S05, ""),
 (r"^(name = tools\.delegate|hits = tools\.grep|page = tools\.read)|is asked to read each item it names", "nonfactual", "", ""),
 (r"best fit and at most one fallback|best fit first and at most one fallback|names the best fit|names the one that fits best|stored item fits your current request best|the librarian's answer for the current request|^When: Call it once per question|^# question\): the best fit", "checked", S07, ""),
 (r"is not run by any call|acts through its episode's|acts through the tools of the episode|from here \(its card says the same\)|^<path> does not exist in the library|^Library files: `/memory-library`|^\| WrongCall|^not_in_library:", "checked", S05, ""),
 (r"^/inputs/episodes/<episode>\.json|/inputs/blobs/<id>", "checked", S13E, ""),
 (r"bounded summary with no code|bounded and with no code|whole code\), and the card ends|docstring and the names of its functions|^An :class:`Item` is a frozen record|^A procedure's card shows the one call|shows on its card the one call|^``'captures'``; None|^\| ``print\(item\)`` shows its card", "checked", S11C, ""),
 (r"^A capture is a library entry from the moment|^Every item's brackets also say|^`memory\.get` opens an item|The harness runs no capture|^\| With no arguments, a procedure that takes the request|^\| \(its refusal is its own|its later uses|use record", "checked", S01, ""),
 (r"opens at most 3 librarian sessions", "checked", J3N, ""),
 (r"each in a fresh conversation", "checked", S11F, ""),
 (r"^`memory\.items` browses every function, note and recorded procedure", "checked", S11K, ""),
 (r"^New captures come to you as a group of related captures|^With them come related captures closed earlier", "checked", S02, ""),
 (r"^Beside it, /inputs/captures/<id>/|^Open entries listed beside them|^Turn a captured attempt|^When: A capture becomes library items", "checked", S13B, ""),
 (r"re-derives every copied fixture|re-derives each copied one", "checked", FIXT, "a unit-level checker (the builder's): no system test makes a writer copy a fixture"),
 (r"^failed \(list of string|^ask_library\(question, schema=None, failed=None\)", "checked", S04, ""),
 (r"^Read a file or list a directory under /inputs, /task or /memory|^Search files under a path of /inputs, /task|^grep\(pattern, path=None, offset=None\) - search files with a regular expression \(default under /inputs; also|^read\(path, offset=None\) - read a file or list a directory under /inputs, /task", "checked", S13A, ""),
 (r"^<path> is outside /workspace|^<path> does not exist; tools\.read\('/memory'\)|^not_found: <path> does not exist; tools\.read|^<path> is outside /inputs, /task, /memory and /work|^outside_box: <path> is outside /inputs, /task", "checked", S13C, ""),
]
ROUND3 = [
 (r"no librarian role", "untestable", "", "a run with memory always seeds the librarian role, so this refusal cannot fire"),
 (r"not a capture of the store|failed or retired attempt|land_capture|^Returns: JSON text: \{\"scope\"|Taking a closed entry|^Without episode", "checked", S13B, ""),
 (r"^(Your box: /inputs \(read-only\); /task|/memory holds the library|They also read /task|The library under /memory may be empty|`execute_code` runs Python in your own box|`memory` \(the library under /memory\)|What /inputs/staging and /inputs/pool|`tools\.read` and `tools\.grep` read /inputs|Search files under a path \(default /inputs\)|grep\(pattern, path=None, offset=None\) - search files with a regular expression \(default under /inputs\))", "checked", S13A, ""),
 (r"is outside /inputs, /memory and /outputs|is outside the library, /memory|^read\(path, offset=None\) - read a file or list a directory under /inputs|^Read a file or list a directory under /inputs, /memory or /outputs", "checked", S13C, ""),
 (r"later librarian of this run reads every note|^Rank every item of the library|`tools\.search` lists every item|procedure's card in a search result shows the call|^``module:name``, a note's slug", "checked", S13D, ""),
]
FALSE = [
 (r"/inputs/attention/index\.json lists your entries' material", "A16", S06+" "+S05, "false for the actor, teammates and librarian (a memory-off run, and paths they cannot open); true for the writer"),
 (r"memory\.ask\(\) shows the ones that apply", "A14", S07, ""),
 (r"every item (it|the librarian) judge[sd]? applicable", "A21", S07, ""),
 (r"exec\(memory\.get\(", "A19", S05, "exec is not a builtin in actor or teammate cells"),
 (r"a captured procedure, or a procedure with a recipe\)", "A24", S05, "a state_env capture refuses a no-argument call"),
 (r"^finish\(answer\) - end the pass; the gate judges", "L1", J2, "the librarian's finish hands an answer back"),
 (r"curator", "L1", J2, "there is no curator"),
 (r"The harness has already run each procedure", "W1", J3, "nothing runs a capture"),
 (r"A capture is not in the library", "W2", J2, "0R: a capture is servable at once"),
 (r"A failed capture is knowledge too", "W4", S12, "steering"),
 (r"The gate runs the function on the recorded bytes again", "W19", J3, "the gate runs no code"),
 (r"The gate runs a procedure's recipe on its cases", "W18", J3, "the gate runs no code"),
 (r"whose row says it reproduces", "W21", J2, "no row carries reproduces"),
 (r"with the output of any failing test", "W23", J2, "the gate runs no test"),
 (r"same_as|gives the same value on its r", "W20", J2, "nothing computes same_as"),
 (r"reads each item it names", "A3", "", "unverifiable: nothing logs the librarian's reads"),
 (r"/inputs, /memory or /outputs|outside /inputs, /memory and /work", "HL2", J2, "false for the writer's helpers (their roots are /inputs, /task, /memory, /work); true for the writer"),
]
NONFACT = r"^(When:|#|page =|hits =|print\(|memory\.ask\(|<pkg>|Search first|Name a second|Rank by fit|Never store|Keep a note|If the gate refuses|Then read|Check the manifest|Run the tests|For each item that applies|Open entries listed|Append one note|You compare recorded|You are the librarian|path \(string|manifest \(string|capture_id \(string|cases \(list|schema \(object|The manifest is /memory/.pass)"
WRITER_FOREIGN = r"^(outside_box: <path> is outside /inputs, /task|outside_library: <path> is outside the library|not_in_library:)"
WRITER_FOREIGN_NOTE = ("true for the role the form belongs to (helper, librarian, actor); false in the writer's tools.help, "
                       "which lists forms the writer is never refused with")


def classify(s: str, roles: list[str], topics: list[str]) -> tuple[str, str, str, str]:
    """(status, checker, audit id, note) of one sentence shown to *roles*."""
    if "writer" in roles and re.search(WRITER_FOREIGN, s):
        return "false", S13C, "HL2w", WRITER_FOREIGN_NOTE
    for rx, aid, chk, nt in FALSE:
        if aid == "A16" and not ({"actor", "teammate", "librarian"} & set(roles)):
            continue  # the writer's own next step: true for the writer
        if aid == "HL2" and "helper" not in roles:
            continue  # the writer's own roots: true for the writer
        if re.search(rx, s):
            return "false", chk, aid, nt
    for rx, st, chk, nt in ROUND4 + ROUND3:
        if re.search(rx, s):
            return st, chk, "", nt
    rules = [
        (re.search(NONFACT, s), "nonfactual", ""),
        (set(roles) <= {"actor", "teammate"} and re.search(r"/memory", s), "checked", S05),
        (set(roles) <= {"actor", "teammate"} and "paths" in topics, "checked", S06),
        ("off" in topics or re.search(r"^(Every agent of this run|The librarian answers|Post a message|Who is on the run)", s),
         "checked", S06),
        ("gate" in topics and re.search(r"runs no|runs neither|runs no code", s), "checked", J3),
        ("gate" in topics and re.search(r"refuse|label|Only these|judges|checks", s), "checked", S09),
        ("gate" in topics and re.search(r"Decisions take effect", s), "checked", S04R),
        (roles == ["memory_help"] and "ask" in topics, "checked", S07),
        (roles == ["memory_help"] and "cards" in topics, "checked", S01),
        (set(roles) <= {"actor", "teammate"} and "ask" in topics, "checked", S07),
        (set(roles) <= {"actor", "teammate", "cards"} and "cards" in topics, "checked", S01),
        (roles == ["librarian"] and "paths" in topics, "checked", S05),
        ("writer" in roles and "paths" in topics and "helper" not in roles, "checked", S05),
    ]
    for hit, status, checker in rules:
        if hit:
            return status, checker, "", ""
    return "unchecked", "", "", ""


rows = []
for s, v in sorted(sc.items()):
    roles = sorted({k.split("/")[0] for k in v["keys"]})
    status, checker, audit, note = classify(s, roles, v["topics"])
    rows.append((status, checker, audit, note, ",".join(roles), s))
with open(sys.argv[3], "w") as f:
    f.write("# The claims registry (guard J1): every in-scope sentence an agent is shown (render.scoped), reviewed.\n")
    f.write("# status: checked (a test exercises it) | false (known false: audit id, the test that fails on it) | "
            "nonfactual (an instruction, example or format) | unchecked (factual, no checker yet)\n")
    f.write("# A new or changed sentence fails test_j1_claims until it is added here after review.\n")
    f.write("status\tchecker\taudit\tnote\troles\tsentence\n")
    for r in rows:
        f.write("\t".join(x.replace("\t", " ") for x in r) + "\n")
print(collections.Counter(r[0] for r in rows))
for r in rows:
    if r[0] == "unchecked":
        print("U |", r[4], "|", r[5][:150])
