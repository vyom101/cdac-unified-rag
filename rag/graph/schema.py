"""Domain-derived KG schema for C-DAC website content."""

ENTITY_TYPES = {
    "Organization", "Division", "Product", "Service", "Location",
    "Person", "Role", "Program", "Technology", "Page",
}
RELATIONSHIP_TYPES = {
    "PART_OF": ("*", "Organization"),
    "LOCATED_IN": ("*", "Location"),
    "OFFERS": ("Organization", "Service"),
    "DEVELOPS": ("Organization", "Product"),
    "HAS_ROLE": ("Person", "Role"),
    "RELATED_TO": ("*", "*"),
    "MENTIONS": ("Page", "*"),
}


def is_valid_entity_type(entity_type: str) -> bool:
    return entity_type in ENTITY_TYPES


def is_valid_relationship(rel_type: str, source_type: str, target_type: str) -> bool:
    if rel_type not in RELATIONSHIP_TYPES:
        return False
    expected_source, expected_target = RELATIONSHIP_TYPES[rel_type]
    return (
        (expected_source == "*" or expected_source == source_type)
        and (expected_target == "*" or expected_target == target_type)
    )
