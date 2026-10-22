"""The literal rule's threshold as the writer's brief states it (:data:`LINT_MIN`): a function may not hard-code a
text of at least this many characters, or a number with at least 2 significant digits, that varies across its
recorded inputs. The gate does not run the lint (rev 2.1, :mod:`.gate_policy`); the brief states the rule.
"""

from __future__ import annotations

LINT_MIN = 4
