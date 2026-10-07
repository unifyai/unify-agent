# PREREG office v1.2, amendment 1

Frozen 2026-10-07T09:17Z by MAIN. Written before any r2 cell, after r1 of both cleanslate arms had finished.

## 1. The r1 full-no-offers exit-1 cleanup flag is cleared; H1 did not fire

The evidence is EVAL/CLEANSLATE, BOARD 09:16Z.

- The kit's survivors() check matches host processes by recorded pid-namespace ids (pid:[inode]).
- Those ids are recycled once a namespace is freed. On w105, 300 sequential bwrap sandboxes produced only 39 distinct ids.
- All 24 workspaces of the cell closed without ConfinementError, and task_dir_removed is true for every one.
- No process carrying the cell's own marker remained.
- The 4 flagged pids belonged to other sandboxes that happened to get recycled namespace ids.

That is a false positive of an over-broad check, not a sign of a confinement problem. The r1 full arm's records stand,
and the kit finding is recorded for a later kit version.

## 2. The rule for remaining cells

A cleanup failure counts as a kit false positive, logged and not H1, only when **all** of the following hold:
- (a) every workspace closed without ConfinementError;
- (b) every task_dir was removed;
- (c) no process carrying the cell's own marker remains;
- (d) the only matches are by pid-namespace id.

Any other cleanup failure, and any other sign of a confinement problem, is H1 as written.

## 3. What is unchanged

Everything else is unchanged: the frozen kit, the source digest 190ddd84…, the arms, the caps and the criteria.
