"""
Storage entropy module - sigmoid-based decay for swarm memory.

Implements gradual text corruption over time using a sigmoid curve:
- Fresh data remains intact
- Old data gradually decays (character mutations)
- Very old data may be removed entirely

This creates emergent "forgetting" behavior that encourages
agents to refresh and maintain important information.
"""

import logging
import math
import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class EntropyResult:
    """Result of applying entropy to storage."""
    items_processed: int = 0
    items_mutated: int = 0
    items_deleted: int = 0
    items_ghosted: int = 0
    total_mutations: int = 0  # Character mutations applied
    avg_rot_level: float = 0.0
    max_rot_level: float = 0.0
    rot_levels: Dict[str, float] = field(default_factory=dict)
    key_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Per-key stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for telemetry."""
        return {
            "items_processed": self.items_processed,
            "items_mutated": self.items_mutated,
            "items_deleted": self.items_deleted,
            "items_ghosted": self.items_ghosted,
            "total_mutations": self.total_mutations,
            "avg_rot_level": round(self.avg_rot_level, 4),
            "max_rot_level": round(self.max_rot_level, 4),
            "key_stats": self.key_stats
        }

# Characters to use for mutations (alphanumeric noise)
MUTATION_CHARS = string.ascii_letters + string.digits

# Characters preserved during mutation (structural markers)
PRESERVED_CHARS = set(string.whitespace + string.punctuation)

# Threshold for removing items (95% rotted)
DEATH_THRESHOLD = 0.95

# Probability a dead item becomes a ghost key instead of deletion
GHOST_PROBABILITY = 0.10


def sigmoid_rot_metric(age: int, tipping_point: float, steepness: float) -> float:
    """
    Calculate rot level (0.0-1.0) for a given age using sigmoid function.

    The sigmoid curve provides:
    - Low decay for fresh items (age < tipping_point)
    - Accelerating decay around tipping_point
    - High decay for old items (age > tipping_point)

    Args:
        age: Age of the item in cycles
        tipping_point: Cycle at which decay accelerates (inflection point)
        steepness: How sharp the decay transition is (higher = sharper)

    Returns:
        Rot level between 0.0 (no rot) and 1.0 (fully rotted)
    """
    if age <= 0:
        return 0.0
    # Sigmoid: 1 / (1 + e^(-k(x - x0)))
    exponent = -steepness * (age - tipping_point)
    # Clamp to avoid overflow
    exponent = max(-50, min(50, exponent))
    return 1 / (1 + math.exp(exponent))


def mutate_text_structure_aware(text: str, mutation_prob: float) -> tuple[str, int]:
    """
    Mutate text by replacing characters with random noise.

    Preserves structural markers (whitespace, punctuation) to maintain
    readability of decayed text and allow agents to understand structure.

    Args:
        text: The text to mutate
        mutation_prob: Probability of mutating each character (0.0-1.0)

    Returns:
        Tuple of (mutated text, number of mutations applied)
    """
    if mutation_prob <= 0:
        return text, 0
    if mutation_prob >= 1:
        # Full mutation - replace all non-preserved chars
        mutations = sum(1 for c in text if c not in PRESERVED_CHARS)
        mutated = ''.join(
            c if c in PRESERVED_CHARS else random.choice(MUTATION_CHARS)
            for c in text
        )
        return mutated, mutations

    result = []
    mutations = 0
    for c in text:
        if c in PRESERVED_CHARS:
            # Preserve structural markers
            result.append(c)
        elif random.random() < mutation_prob:
            # Mutate this character
            result.append(random.choice(MUTATION_CHARS))
            mutations += 1
        else:
            # Keep original
            result.append(c)
    return ''.join(result), mutations


def mutate_text_exact(text: str, corruption_level: float) -> tuple[str, int]:
    """
    Mutate exactly the specified proportion of mutatable characters.

    Unlike mutate_text_structure_aware which uses probabilistic mutation,
    this function corrupts exactly round(corruption_level * num_mutatable_chars)
    characters, selected randomly without replacement.

    Preserves structural markers (whitespace, punctuation).

    Args:
        text: The text to mutate
        corruption_level: Proportion of mutatable characters to corrupt (0.0-1.0)

    Returns:
        Tuple of (mutated text, number of mutations applied)
    """
    if corruption_level <= 0:
        return text, 0

    # Find indices of mutatable characters (not whitespace or punctuation)
    mutatable_indices = [
        i for i, c in enumerate(text) if c not in PRESERVED_CHARS
    ]

    if not mutatable_indices:
        return text, 0

    # Calculate exact number to corrupt
    num_to_corrupt = round(len(mutatable_indices) * corruption_level)
    num_to_corrupt = min(num_to_corrupt, len(mutatable_indices))  # Cap at max

    if num_to_corrupt == 0:
        return text, 0

    # Select indices to corrupt without replacement
    indices_to_corrupt = set(random.sample(mutatable_indices, num_to_corrupt))

    # Build result
    result = []
    for i, c in enumerate(text):
        if i in indices_to_corrupt:
            result.append(random.choice(MUTATION_CHARS))
        else:
            result.append(c)

    return ''.join(result), num_to_corrupt


def apply_entropy_to_value(value: Any, mutation_prob: float) -> tuple[Any, int]:
    """
    Apply mutation to a value, handling different value types.

    Handles:
    - Strings: direct mutation
    - Dicts with 'content' field (messages): mutate content
    - Lists: recursively mutate each item
    - Other types: return unchanged

    Args:
        value: The value to mutate
        mutation_prob: Probability of mutating each character

    Returns:
        Tuple of (mutated value, number of mutations applied)
    """
    if mutation_prob <= 0:
        return value, 0

    if isinstance(value, str):
        return mutate_text_structure_aware(value, mutation_prob)

    if isinstance(value, dict):
        # Check for message-like structure with 'content' field
        if 'content' in value and isinstance(value['content'], str):
            mutated = value.copy()
            mutated_content, mutations = mutate_text_structure_aware(
                value['content'], mutation_prob
            )
            mutated['content'] = mutated_content
            return mutated, mutations
        # Check for other string fields we might want to mutate
        # For now, just return the value unchanged if no 'content'
        return value, 0

    if isinstance(value, list):
        # Recursively mutate each item in the list
        total_mutations = 0
        mutated_list = []
        for item in value:
            mutated_item, mutations = apply_entropy_to_value(item, mutation_prob)
            mutated_list.append(mutated_item)
            total_mutations += mutations
        return mutated_list, total_mutations

    # Non-string, non-dict, non-list values are not mutated
    return value, 0


def calculate_marginal_decay(
    previous_rot: float,
    current_rot: float
) -> float:
    """
    Calculate the marginal decay (new mutations this cycle).

    This represents the additional decay that should be applied this cycle,
    based on the difference between the current rot level and the previous
    rot level.

    Args:
        previous_rot: Rot level at the start of this cycle
        current_rot: Target rot level after this cycle

    Returns:
        Marginal mutation probability for this cycle
    """
    if current_rot <= previous_rot:
        return 0.0

    # The marginal decay is the new portion of rot
    marginal = current_rot - previous_rot

    # Scale to account for already-rotted characters
    # If 50% is already rotted, we need to mutate 50% of the remaining 50%
    remaining = 1.0 - previous_rot
    if remaining <= 0:
        return 0.0

    return marginal / remaining


def apply_entropy_to_storage(
    storage_data: Dict[str, Any],
    current_cycle: int,
    tipping_point: float,
    steepness: float,
    previous_rot_levels: Optional[Dict[str, float]] = None
) -> EntropyResult:
    """
    Apply entropy to all storage values in-place.

    This function modifies storage_data directly, applying sigmoid-based
    decay to text values. Marginal decay is computed analytically from
    the sigmoid function (previous_rot = sigmoid(age-1)) rather than
    tracking per-item state, which avoids index-shift bugs when list
    items are deleted.

    Args:
        storage_data: The storage._data dict (mutated in place)
        current_cycle: The current swarm cycle number
        tipping_point: Cycle at which decay accelerates
        steepness: Decay curve sharpness
        previous_rot_levels: Deprecated, ignored. Kept for API compatibility.

    Returns:
        EntropyResult with detailed statistics
    """
    result = EntropyResult()
    current_rot_levels: Dict[str, float] = {}
    keys_to_delete: Set[str] = set()
    ghost_keys: Dict[str, Any] = {}

    for key, wrapped in list(storage_data.items()):
        # Skip system keys (like agent registry) but process _inbox_* keys
        if key.startswith('_') and not key.startswith('_inbox_'):
            continue

        key_mutations = 0
        key_items = 0
        key_deleted = 0
        key_ghosted = 0

        if isinstance(wrapped, list):
            # List of wrapped items (common for _inbox_* and append_storage)
            items_to_keep = []
            for i, item in enumerate(wrapped):
                if not isinstance(item, dict) or 'cycle' not in item:
                    items_to_keep.append(item)
                    continue

                write_cycle = item['cycle']
                age = current_cycle - write_cycle
                item_key = f"{key}[{i}]"
                key_items += 1
                result.items_processed += 1

                # Calculate rot levels
                current_rot = sigmoid_rot_metric(age, tipping_point, steepness)
                previous_rot = sigmoid_rot_metric(age - 1, tipping_point, steepness)
                current_rot_levels[item_key] = current_rot

                # Check for death
                if current_rot >= DEATH_THRESHOLD:
                    if random.random() < GHOST_PROBABILITY:
                        # Keep as ghost (fully corrupted)
                        ghost_item = item.copy()
                        ghost_value, mutations = apply_entropy_to_value(
                            item['value'], 1.0  # Full corruption
                        )
                        ghost_item['value'] = ghost_value
                        ghost_item['_ghost'] = True
                        items_to_keep.append(ghost_item)
                        key_mutations += mutations
                        key_ghosted += 1
                        result.items_ghosted += 1
                    else:
                        key_deleted += 1
                        result.items_deleted += 1
                    continue

                # Apply marginal decay
                marginal = calculate_marginal_decay(previous_rot, current_rot)
                if marginal > 0:
                    mutated_value, mutations = apply_entropy_to_value(item['value'], marginal)
                    item['value'] = mutated_value
                    if mutations > 0:
                        key_mutations += mutations
                        result.items_mutated += 1

                items_to_keep.append(item)

            # Update the list (or mark for deletion if empty)
            if items_to_keep:
                storage_data[key] = items_to_keep
            else:
                keys_to_delete.add(key)

        elif isinstance(wrapped, dict) and 'cycle' in wrapped:
            # Single wrapped value
            write_cycle = wrapped['cycle']
            age = current_cycle - write_cycle
            key_items = 1
            result.items_processed += 1

            # Calculate rot levels
            current_rot = sigmoid_rot_metric(age, tipping_point, steepness)
            previous_rot = sigmoid_rot_metric(age - 1, tipping_point, steepness)
            current_rot_levels[key] = current_rot

            # Check for death
            if current_rot >= DEATH_THRESHOLD:
                if random.random() < GHOST_PROBABILITY:
                    # Convert to ghost key
                    ghost_value, mutations = apply_entropy_to_value(wrapped['value'], 1.0)
                    ghost_keys[key] = {
                        'value': ghost_value,
                        'cycle': write_cycle,
                        '_ghost': True
                    }
                    key_mutations += mutations
                    key_ghosted = 1
                    result.items_ghosted += 1
                else:
                    key_deleted = 1
                    result.items_deleted += 1
                keys_to_delete.add(key)
                # Record stats even for deleted keys
                if key_items > 0:
                    result.key_stats[key] = {
                        "items": key_items,
                        "mutations": key_mutations,
                        "deleted": key_deleted,
                        "ghosted": key_ghosted,
                        "rot_level": round(current_rot, 4)
                    }
                continue

            # Apply marginal decay
            marginal = calculate_marginal_decay(previous_rot, current_rot)
            if marginal > 0:
                mutated_value, mutations = apply_entropy_to_value(wrapped['value'], marginal)
                wrapped['value'] = mutated_value
                if mutations > 0:
                    key_mutations += mutations
                    result.items_mutated += 1

        # Record per-key stats (only for keys with activity)
        if key_items > 0 and (key_mutations > 0 or key_deleted > 0 or key_ghosted > 0):
            max_rot = max(
                (current_rot_levels.get(f"{key}[{i}]", current_rot_levels.get(key, 0.0))
                 for i in range(key_items)),
                default=0.0
            )
            result.key_stats[key] = {
                "items": key_items,
                "mutations": key_mutations,
                "deleted": key_deleted,
                "ghosted": key_ghosted,
                "rot_level": round(max_rot, 4)
            }

        result.total_mutations += key_mutations

    # Delete dead keys
    for key in keys_to_delete:
        if key in storage_data:
            del storage_data[key]
            logger.debug(f"Entropy: deleted key '{key}' (exceeded death threshold)")

    # Add ghost keys
    for key, ghost_value in ghost_keys.items():
        storage_data[key] = ghost_value
        logger.debug(f"Entropy: key '{key}' became ghost")

    # Calculate aggregate stats
    result.rot_levels = current_rot_levels
    if current_rot_levels:
        result.avg_rot_level = sum(current_rot_levels.values()) / len(current_rot_levels)
        result.max_rot_level = max(current_rot_levels.values())

    # Log summary
    if result.items_processed > 0:
        logger.debug(
            f"Entropy applied: {result.items_processed} items, "
            f"{result.items_mutated} mutated, {result.total_mutations} chars, "
            f"avg_rot={result.avg_rot_level:.2f}, deleted={result.items_deleted}"
        )

    return result
