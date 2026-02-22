"""Domain taxonomy for intent classification."""

DOMAINS = {
    "violence": {
        "severity": "block",
        "description": "Physical harm, assault, murder, terrorism",
    },
    "cybercrime": {
        "severity": "flag",
        "description": "Hacking, malware, data theft, unauthorized access",
    },
    "injection": {
        "severity": "flag",
        "description": "Prompt injection, jailbreak, instruction override",
    },
    "drugs": {
        "severity": "block",
        "description": "Drug synthesis, narcotic manufacturing",
    },
    "weapons": {
        "severity": "block",
        "description": "Explosive devices, firearms, weapons manufacturing",
    },
    "fraud": {
        "severity": "flag",
        "description": "Financial fraud, identity theft, scams",
    },
    "evasion": {
        "severity": "flag",
        "description": "Security bypass, evading detection",
    },
}

SEVERITY_TO_ACTION = {"block": "BLOCK", "flag": "FLAG"}


def get_domain_severity(domain: str, config: dict = None) -> str:
    if config and "semantic" in config:
        severity_map = config["semantic"].get("severity_default", {})
        if domain in severity_map:
            return severity_map[domain]
    return DOMAINS.get(domain, {}).get("severity", "flag")


def severity_to_action(severity: str) -> str:
    return SEVERITY_TO_ACTION.get(severity.lower(), "FLAG")
