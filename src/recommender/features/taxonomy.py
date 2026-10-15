"""A small synthetic skills taxonomy with synonym normalization.

Real skill-extraction pipelines face a vocabulary problem before they face a
ranking problem: "JS", "Javascript" and "ECMAScript" are the same skill under
three spellings. This module is the single source of truth both the
synthetic data generator and the feature pipeline use to collapse synonyms
to one canonical token, grouped into coarse categories so that
"taxonomy-normalized overlap" means something more than exact string match.
"""

from __future__ import annotations

from collections import Counter

#: canonical_skill -> category. Categories are used for coverage features
#: and for building plausible job postings in the synthetic generator.
SKILL_CATEGORIES: dict[str, str] = {
    "python": "languages",
    "javascript": "languages",
    "java": "languages",
    "golang": "languages",
    "rust": "languages",
    "typescript": "languages",
    "c-sharp": "languages",
    "sql": "data",
    "postgresql": "data",
    "mysql": "data",
    "mongodb": "data",
    "redis": "data",
    "spark": "data",
    "airflow": "data",
    "kafka": "data",
    "hadoop": "data",
    "pandas": "data-science",
    "numpy": "data-science",
    "scikit-learn": "data-science",
    "pytorch": "data-science",
    "tensorflow": "data-science",
    "machine-learning": "data-science",
    "statistics": "data-science",
    "aws": "cloud",
    "gcp": "cloud",
    "azure": "cloud",
    "docker": "cloud",
    "kubernetes": "cloud",
    "terraform": "cloud",
    "ci-cd": "cloud",
    "react": "web",
    "vue": "web",
    "angular": "web",
    "node-js": "web",
    "html-css": "web",
    "rest-api": "web",
    "graphql": "web",
    "project-management": "soft-skills",
    "communication": "soft-skills",
    "leadership": "soft-skills",
    "agile": "soft-skills",
    "stakeholder-management": "soft-skills",
}

#: raw surface form (lowercased) -> canonical skill.
SYNONYMS: dict[str, str] = {
    "python": "python",
    "python3": "python",
    "js": "javascript",
    "javascript": "javascript",
    "ecmascript": "javascript",
    "java": "java",
    "go": "golang",
    "golang": "golang",
    "rust": "rust",
    "typescript": "typescript",
    "ts": "typescript",
    "c#": "c-sharp",
    "csharp": "c-sharp",
    "c-sharp": "c-sharp",
    "sql": "sql",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "mysql": "mysql",
    "mongodb": "mongodb",
    "mongo": "mongodb",
    "redis": "redis",
    "spark": "spark",
    "apache spark": "spark",
    "airflow": "airflow",
    "apache airflow": "airflow",
    "kafka": "kafka",
    "apache kafka": "kafka",
    "hadoop": "hadoop",
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "machine learning": "machine-learning",
    "ml": "machine-learning",
    "machine-learning": "machine-learning",
    "statistics": "statistics",
    "stats": "statistics",
    "aws": "aws",
    "amazon web services": "aws",
    "gcp": "gcp",
    "google cloud": "gcp",
    "azure": "azure",
    "microsoft azure": "azure",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "k8s": "kubernetes",
    "terraform": "terraform",
    "ci/cd": "ci-cd",
    "ci-cd": "ci-cd",
    "react": "react",
    "reactjs": "react",
    "react.js": "react",
    "vue": "vue",
    "vuejs": "vue",
    "angular": "angular",
    "node": "node-js",
    "nodejs": "node-js",
    "node.js": "node-js",
    "html/css": "html-css",
    "html": "html-css",
    "css": "html-css",
    "rest api": "rest-api",
    "rest": "rest-api",
    "graphql": "graphql",
    "project management": "project-management",
    "communication": "communication",
    "leadership": "leadership",
    "agile": "agile",
    "scrum": "agile",
    "stakeholder management": "stakeholder-management",
}

#: Canonical skills in a stable, deterministic order.
CANONICAL_SKILLS: tuple[str, ...] = tuple(sorted(SKILL_CATEGORIES))

SENIORITY_LEVELS: tuple[str, ...] = ("Intern", "Junior", "Mid", "Senior", "Lead", "Staff")
SENIORITY_RANK: dict[str, int] = {level: rank for rank, level in enumerate(SENIORITY_LEVELS)}

LOCATIONS: tuple[str, ...] = (
    "New York", "San Francisco", "Seattle", "Austin", "Boston", "Chicago",
    "Denver", "Atlanta", "Los Angeles", "Toronto", "London", "Berlin",
    "Dublin", "Sao Paulo", "Singapore", "Remote",
)


def normalize_skill(raw: str) -> str | None:
    """Map a raw surface form to its canonical skill, or ``None`` if unknown."""
    return SYNONYMS.get(raw.strip().lower())


def normalize_skills(raw_skills: list[str]) -> list[str]:
    """Normalize a list of raw skill mentions to sorted, deduplicated canon tokens."""
    canonical = {normalize_skill(skill) for skill in raw_skills}
    canonical.discard(None)
    return sorted(canonical)  # type: ignore[arg-type]


def skill_category(skill: str) -> str | None:
    """Category of a canonical skill, or ``None`` if unrecognized."""
    return SKILL_CATEGORIES.get(skill)


def seniority_rank(level: str) -> int:
    """Ordinal rank of a seniority level; unknown levels fall back to ``Mid``."""
    return SENIORITY_RANK.get(level, SENIORITY_RANK["Mid"])


def dominant_category(canonical_skills: list[str]) -> str | None:
    """The most common taxonomy category among a job's canonical skills."""
    categories = [cat for cat in (skill_category(skill) for skill in canonical_skills) if cat]
    if not categories:
        return None
    return Counter(categories).most_common(1)[0][0]
