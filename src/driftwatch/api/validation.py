from typing import Any

from driftwatch.config.schema import ModelSchemaSpec


def validate_features(features: dict[str, Any], schema: ModelSchemaSpec) -> list[str]:
    errors: list[str] = []
    declared = {feature.name: feature for feature in schema.features}

    unknown = sorted(set(features) - set(declared))
    if unknown:
        errors.append(f"unknown features: {unknown}")

    for name, spec in declared.items():
        value = features.get(name)
        if value is None:
            if not spec.nullable:
                errors.append(f"feature {name!r} is required")
            continue
        if spec.dtype == "continuous":
            if isinstance(value, bool) or not isinstance(value, int | float):
                errors.append(f"feature {name!r} must be numeric")
        elif spec.dtype == "categorical" and not isinstance(value, str | bool):
            errors.append(f"feature {name!r} must be a string or boolean")

    return errors


def extract_segment_values(features: dict[str, Any], dimensions: list[str]) -> dict[str, str]:
    return {
        dimension: str(features[dimension])
        for dimension in dimensions
        if features.get(dimension) is not None
    }
