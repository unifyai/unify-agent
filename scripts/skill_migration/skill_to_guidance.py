"""Core library for importing ``SKILL.md`` skills into the GuidanceManager.

OpenClaw and HermesAgent both represent skills with the agentskills.io
standard: a ``SKILL.md`` file carrying YAML frontmatter (``name``,
``description``, ...) followed by a markdown body of procedural
instructions, optionally accompanied by bundled ``scripts/`` helpers.

Unify's :class:`~unify.guidance_manager.guidance_manager.GuidanceManager`
stores the same kind of procedural how-to content as guidance entries
(``title`` + freeform ``content``).  The mapping is therefore close to
one-to-one:

    skill ``name``                  -> guidance ``title`` (namespaced)
    skill ``description`` + body    -> guidance ``content``
    bundled ``scripts/`` files      -> inlined verbatim into ``content``

Bundled scripts are inlined as fenced code blocks rather than executed or
registered.  This keeps the import a pure textual transfer; lifting any of
that code into the FunctionManager and linking it back via ``function_ids``
is a deliberate, separate step.

The parsing / mapping helpers here are pure (no backend, no LLM) so they
can be unit-tested cheaply.  :class:`SkillMigrator` layers the actual
GuidanceManager writes on top.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Title column is capped at 200 chars by the Guidance schema.
MAX_TITLE_CHARS = 200
# Per-script inline cap, so a single huge helper can't dominate an entry.
DEFAULT_MAX_SCRIPT_CHARS = 20000
# Files under a skill's ``scripts/`` dir with these suffixes are inlined as text.
TEXT_SCRIPT_SUFFIXES = {
    ".sh",
    ".bash",
    ".zsh",
    ".py",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".rb",
    ".pl",
    ".ps1",
    ".lua",
    ".r",
    ".sql",
    "",  # extensionless executables (shebang scripts)
}
# Map suffixes to markdown fence languages for readable inlining.
_FENCE_LANG = {
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".rb": "ruby",
    ".pl": "perl",
    ".ps1": "powershell",
    ".lua": "lua",
    ".r": "r",
    ".sql": "sql",
}


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #


def split_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Split a ``SKILL.md`` document into (frontmatter dict, markdown body).

    Frontmatter is the optional ``---`` fenced YAML block at the very top of
    the file.  OpenClaw stores its ``metadata`` as a single-line JSON object
    and HermesAgent uses nested YAML; both are valid YAML, so a single
    ``yaml.safe_load`` handles them uniformly.  When no frontmatter is
    present the whole document is treated as the body.
    """
    if not text.startswith("---"):
        return {}, text.lstrip("\n")

    lines = text.splitlines()
    close_idx: Optional[int] = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx is None:
        # Unterminated fence — treat the document as plain body.
        return {}, text.lstrip("\n")

    fm_text = "\n".join(lines[1:close_idx])
    body = "\n".join(lines[close_idx + 1 :]).lstrip("\n")
    try:
        parsed = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        logger.warning("Failed to parse YAML frontmatter; treating as empty")
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed, body


@dataclass
class ParsedSkill:
    """A single skill parsed from a ``SKILL.md`` file."""

    name: str
    description: str
    body: str
    frontmatter: Dict[str, Any]
    source: str
    rel_path: str
    skill_dir: Path
    scripts: List[Path] = field(default_factory=list)
    references: List[Path] = field(default_factory=list)


def _collect_supporting_files(skill_dir: Path, subdir: str) -> List[Path]:
    root = skill_dir / subdir
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


def parse_skill_file(
    skill_md: Path,
    *,
    source: str,
    repo_root: Path,
) -> ParsedSkill:
    """Parse a single ``SKILL.md`` into a :class:`ParsedSkill`."""
    text = skill_md.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(text)

    skill_dir = skill_md.parent
    name = str(frontmatter.get("name") or skill_dir.name).strip()
    description = str(frontmatter.get("description") or "").strip()

    try:
        rel_path = str(skill_md.relative_to(repo_root))
    except ValueError:
        rel_path = str(skill_md)

    return ParsedSkill(
        name=name,
        description=description,
        body=body,
        frontmatter=frontmatter,
        source=source,
        rel_path=rel_path,
        skill_dir=skill_dir,
        scripts=_collect_supporting_files(skill_dir, "scripts"),
        references=_collect_supporting_files(skill_dir, "references"),
    )


def discover_skills(
    roots: List[Path],
    *,
    source: str,
    repo_root: Path,
) -> List[ParsedSkill]:
    """Find and parse every ``SKILL.md`` beneath any of ``roots``.

    Results are de-duplicated by resolved path and sorted by relative path
    for deterministic ordering.
    """
    seen: set[Path] = set()
    skills: List[ParsedSkill] = []
    for root in roots:
        if not root.exists():
            continue
        for skill_md in sorted(root.rglob("SKILL.md")):
            resolved = skill_md.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            skills.append(
                parse_skill_file(skill_md, source=source, repo_root=repo_root),
            )
    skills.sort(key=lambda s: s.rel_path)
    return skills


# --------------------------------------------------------------------------- #
# Mapping skill -> guidance                                                    #
# --------------------------------------------------------------------------- #


def guidance_title(skill: ParsedSkill, *, title_prefix: str) -> str:
    """Build the guidance title for a skill, namespaced and length-capped."""
    title = f"{title_prefix}{skill.name}".strip()
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS]
    return title


def _fence_lang(path: Path) -> str:
    return _FENCE_LANG.get(path.suffix.lower(), "")


def _inline_scripts_section(
    skill: ParsedSkill,
    *,
    max_script_chars: int,
) -> str:
    text_scripts = [
        p for p in skill.scripts if p.suffix.lower() in TEXT_SCRIPT_SUFFIXES
    ]
    if not text_scripts:
        return ""

    parts: List[str] = [
        "## Bundled scripts (textual reference)",
        (
            "These helper scripts ship with the original skill and are "
            "included verbatim for reference. They are NOT registered as "
            "executable functions by this import — lift any you want to run "
            "into the FunctionManager and link them back to this guidance "
            "where appropriate."
        ),
    ]
    for path in text_scripts:
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; skip from inline text
        if len(content) > max_script_chars:
            content = (
                content[:max_script_chars]
                + f"\n\n... (truncated; {len(content)} chars total)"
            )
        try:
            rel = path.relative_to(skill.skill_dir)
        except ValueError:
            rel = path
        lang = _fence_lang(path)
        parts.append(f"### {rel}\n```{lang}\n{content}\n```")
    if len(parts) == 2:
        return ""  # all candidate scripts were binary/unreadable
    return "\n\n".join(parts)


def _provenance_footer(skill: ParsedSkill) -> str:
    lines = [f"Source: {skill.source} · `{skill.rel_path}`"]

    meta = skill.frontmatter.get("metadata")
    tags: List[str] = []
    if isinstance(meta, dict):
        for vendor in ("hermes", "openclaw"):
            block = meta.get(vendor)
            if isinstance(block, dict) and isinstance(block.get("tags"), list):
                tags.extend(str(t) for t in block["tags"])
    if tags:
        lines.append("Tags: " + ", ".join(tags))

    if skill.references:
        ref_names = ", ".join(sorted(p.name for p in skill.references))
        lines.append(f"Reference docs (not inlined): {ref_names}")

    return "\n".join(lines)


def compose_guidance_content(
    skill: ParsedSkill,
    *,
    inline_scripts: bool = True,
    max_script_chars: int = DEFAULT_MAX_SCRIPT_CHARS,
) -> str:
    """Compose the guidance ``content`` string for a parsed skill.

    The skill ``description`` is folded into the body so it is part of the
    ``content`` that GuidanceManager search embeds and compares with a
    query. Bundled scripts are optionally inlined and a provenance footer
    records where the skill came from.
    """
    sections: List[str] = []
    if skill.description:
        sections.append(skill.description)
    compatibility = skill.frontmatter.get("compatibility")
    if compatibility:
        # Surface environment requirements to the LLM as part of the
        # searchable content rather than dropping the frontmatter field.
        sections.append(f"Compatibility: {str(compatibility).strip()}")
    if skill.body.strip():
        sections.append(skill.body.strip())
    if inline_scripts:
        scripts_section = _inline_scripts_section(
            skill,
            max_script_chars=max_script_chars,
        )
        if scripts_section:
            sections.append(scripts_section)
    sections.append("---\n" + _provenance_footer(skill))

    content = "\n\n".join(sections).strip()
    # Guidance.content requires min_length=1.
    return content or skill.name


# --------------------------------------------------------------------------- #
# Migrator                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class GuidanceEntry:
    """A guidance entry ready to be written, plus its skill provenance."""

    name: str
    title: str
    content: str
    source: str
    rel_path: str


class SkillMigrator:
    """Map discovered skills to guidance entries and (optionally) persist them.

    Parameters
    ----------
    skills:
        Parsed skills to migrate.
    guidance_manager:
        A ``GuidanceManager`` instance.  Required for :meth:`run` with
        ``execute=True``; not needed to :meth:`build_entries` or to produce a
        dry-run report.
    title_prefix:
        Prefixed to every skill name to namespace it (e.g. ``"[openclaw] "``).
        Namespacing avoids silent collisions between the two source repos,
        which share several skill names (arxiv, github, ...).
    conflict_mode:
        ``"skip"`` (leave an existing same-title entry untouched) or
        ``"overwrite"`` (update the existing entry's content in place).
    inline_scripts:
        Whether to inline bundled ``scripts/`` files into the content.
    """

    def __init__(
        self,
        skills: List[ParsedSkill],
        *,
        guidance_manager: Any = None,
        title_prefix: str,
        conflict_mode: str = "skip",
        inline_scripts: bool = True,
        max_script_chars: int = DEFAULT_MAX_SCRIPT_CHARS,
    ):
        if conflict_mode not in ("skip", "overwrite"):
            raise ValueError(
                f"conflict_mode must be 'skip' or 'overwrite', got {conflict_mode!r}",
            )
        self.skills = skills
        self.gm = guidance_manager
        self.title_prefix = title_prefix
        self.conflict_mode = conflict_mode
        self.inline_scripts = inline_scripts
        self.max_script_chars = max_script_chars

    def build_entries(self) -> List[GuidanceEntry]:
        """Pure mapping of skills to guidance entries (no backend access)."""
        entries: List[GuidanceEntry] = []
        for skill in self.skills:
            entries.append(
                GuidanceEntry(
                    name=skill.name,
                    title=guidance_title(skill, title_prefix=self.title_prefix),
                    content=compose_guidance_content(
                        skill,
                        inline_scripts=self.inline_scripts,
                        max_script_chars=self.max_script_chars,
                    ),
                    source=skill.source,
                    rel_path=skill.rel_path,
                ),
            )
        return entries

    def _existing_guidance_id(self, title: str) -> Optional[int]:
        rows = self.gm.filter(filter=f"title == {title!r}", limit=1)
        if not rows:
            return None
        row = rows[0]
        return getattr(row, "guidance_id", None)

    def run(self, *, execute: bool) -> Dict[str, Any]:
        """Migrate skills into the GuidanceManager.

        When ``execute`` is False this is a dry run: titles/content are built
        and existing entries are detected, but nothing is written. When True,
        new entries are created and (in ``overwrite`` mode) conflicting
        entries are updated.
        """
        if execute and self.gm is None:
            raise ValueError("execute=True requires a guidance_manager")

        entries = self.build_entries()
        items: List[Dict[str, Any]] = []
        counts = {"added": 0, "updated": 0, "skipped": 0, "errors": 0}

        for entry in entries:
            item: Dict[str, Any] = {
                "name": entry.name,
                "title": entry.title,
                "source": entry.source,
                "rel_path": entry.rel_path,
            }
            try:
                existing_id = (
                    self._existing_guidance_id(entry.title)
                    if self.gm is not None
                    else None
                )

                if existing_id is not None and self.conflict_mode == "skip":
                    item["status"] = "skipped"
                    item["reason"] = "title already exists"
                    item["guidance_id"] = existing_id
                    counts["skipped"] += 1
                elif existing_id is not None and self.conflict_mode == "overwrite":
                    if execute:
                        self.gm.update_guidance(
                            guidance_id=existing_id,
                            title=entry.title,
                            content=entry.content,
                        )
                    item["status"] = "updated"
                    item["guidance_id"] = existing_id
                    counts["updated"] += 1
                else:
                    if execute:
                        out = self.gm.add_guidance(
                            title=entry.title,
                            content=entry.content,
                        )
                        item["guidance_id"] = out["details"]["guidance_id"]
                    item["status"] = "added"
                    counts["added"] += 1
            except Exception as exc:  # noqa: BLE001 - report, don't abort batch
                logger.exception("Failed to migrate skill %r", entry.name)
                item["status"] = "error"
                item["reason"] = str(exc)
                counts["errors"] += 1
            items.append(item)

        return {
            "source": self.skills[0].source if self.skills else None,
            "executed": execute,
            "title_prefix": self.title_prefix,
            "conflict_mode": self.conflict_mode,
            "summary": {"discovered": len(entries), **counts},
            "items": items,
        }
