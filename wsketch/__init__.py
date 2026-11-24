"""wsketch: single-stroke shape recognition for whiteboards."""

# Model contract. Index order is part of the exported model, do not reorder.
CLASSES = [
    "none",
    "line",
    "arrow",
    "rectangle",
    "ellipse",
    "triangle",
    "diamond",
    "pentagon",
    "hexagon",
    "star",
    "heart",
    "cloud",
    "scribble",
]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

# The data is always labelled with the 13 kinds above. A model may be trained on a
# smaller set: kinds it does not list count as "none", so those strokes stay ink.
CLASS_SETS = {
    "full": CLASSES,
    # for a classroom whiteboard: connectors and basic geometry only
    "edu": ["none", "line", "arrow", "rectangle", "ellipse", "triangle", "diamond", "pentagon", "hexagon"],
}


def kind_to_class(classes: list[str]) -> list[int]:
    """For every kind in CLASSES, its index in `classes` ("none" when it is not listed)."""
    return [classes.index(k) if k in classes else classes.index("none") for k in CLASSES]


N_POINTS = 128
