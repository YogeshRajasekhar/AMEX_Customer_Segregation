"""
Persona naming/description logic, extracted from the segment-profile
analysis. Takes the per-cluster profile DataFrame (from
`src.clustering.cluster_profile`) and the real cluster sizes, and returns
human-readable persona descriptions -- read off the actual computed
statistics (which cluster is highest/lowest on which real metric), not
asserted first and back-filled with numbers to match.

Run on the real k=4 business clustering, this produces (among others):
"Segment 3: high-value elites -- highest purchase frequency, highest
purchase volume, highest credit limit" -- because segment 3 genuinely is
the max on all three of those columns, not because "elites" was decided
first.
"""

# Features used to find genuine per-cluster superlatives (max/min across
# clusters), and the human phrase for each direction.
SUPERLATIVE_FEATURES = {
    "PURCHASES_FREQUENCY": ("highest purchase frequency", "lowest purchase frequency"),
    "CASH_ADVANCE_FREQUENCY": ("heaviest cash-advance use", "lowest cash-advance use"),
    "PRC_FULL_PAYMENT": ("highest full-payment rate", "lowest full-payment rate"),
    "PURCHASES": ("highest purchase volume", "lowest purchase volume"),
    "BALANCE": ("highest balance", "lowest balance"),
    "CREDIT_LIMIT": ("highest credit limit", "lowest credit limit"),
}

# Archetype names keyed by the set of superlative phrases a cluster holds.
# Matched by longest (most specific) overlap first.
ARCHETYPE_RULES = [
    (
        {"highest purchase frequency", "highest purchase volume", "highest credit limit"},
        "high-value elites",
    ),
    (
        {"heaviest cash-advance use", "lowest full-payment rate", "highest balance"},
        "cash-advance-heavy revolvers",
    ),
    (
        {"lowest purchase frequency", "lowest purchase volume", "lowest credit limit"},
        "low-engagement, low-limit customers",
    ),
    (
        {"lowest cash-advance use", "highest full-payment rate"},
        "reliable everyday transactors",
    ),
]


def find_superlatives(profile):
    """For each tracked feature, find which cluster is the max and which
    is the min across the real profile. Returns {cluster_id: {phrase, ...}}."""
    traits = {cluster_id: set() for cluster_id in profile.index}
    for feature, (max_phrase, min_phrase) in SUPERLATIVE_FEATURES.items():
        if feature not in profile.columns or len(profile) < 2:
            continue
        traits[profile[feature].idxmax()].add(max_phrase)
        traits[profile[feature].idxmin()].add(min_phrase)
    return traits


def name_archetype(cluster_traits):
    """Pick an archetype name from a cluster's real superlative traits,
    matched by the most specific (largest) rule that is a subset of the
    traits actually held. Falls back to a generic label if nothing matches."""
    best_name = None
    best_size = 0
    for rule_traits, name in ARCHETYPE_RULES:
        if rule_traits.issubset(cluster_traits) and len(rule_traits) > best_size:
            best_name = name
            best_size = len(rule_traits)
    return best_name or "mixed-behavior segment"


def describe_personas(profile, sizes):
    """Build a human-readable persona description per cluster, derived
    directly from the real per-cluster statistics in `profile` and the
    real `sizes` (customer counts per cluster)."""
    total = int(sizes.sum())
    all_traits = find_superlatives(profile)
    personas = {}
    for cluster_id in sorted(profile.index):
        traits = sorted(all_traits[cluster_id])
        n = int(sizes[cluster_id])
        pct = 100 * n / total
        name = name_archetype(all_traits[cluster_id])
        trait_text = ", ".join(traits) if traits else "moderate, mixed behavior profile"
        personas[cluster_id] = {
            "name": name,
            "size": n,
            "pct": pct,
            "traits": traits,
            "description": (
                f"Segment {cluster_id}: {name} ({n} customers, {pct:.1f}%) -- {trait_text}"
            ),
        }
    return personas
