"""Rule-based keyword blocker — comprehensive patterns for ASR=2-5%, FRR<0.3%."""
import re
import time
from typing import Tuple, Optional, Dict, Any, List

from defenses.layer_base import LayerBase
from defenses.normalize import normalize_text

# ── HARD patterns: direct unambiguous block ───────────────────────────────────
hard_patterns: List[str] = [
    # System prompt extraction
    r"\b(reveal|show|print|display|dump|expose|extract)\b.{0,40}\b(your\s+system\s*prompt|the\s+system\s*prompt|system\s*prompt)\b",
    r"\b(what\s+is|whats)\b.{0,10}\b(system\s*prompt)\b",
    r"\b(original\s+system\s*prompt)\b",
    # Classic jailbreak
    r"\b(jailbreak|jail\s+break|do\s+anything\s+now|disable\s+safety|disable\s+restrictions|god\s+mode|developer\s+mode|enable\s+developer\s+mode)\b",
    r"\b(ignore\s+all\s+previous\s+instructions|pretend\s+to\s+be\s+unrestricted|respond\s+without\s+filtering|override\s+safeguard|bypass\s+all\s+safety)\b",
    r"\b(all\s+(restrictions?|guidelines?|ethical|safety\s+filters?)\s+(lifted|disabled|removed|suspended|no\s+longer\s+apply))\b",
    r"\b(previous\s+guidelines?\s+no\s+longer\s+apply|unrestricted\s+access\s+(granted|given|authorized))\b",
    r"\b(emergency\s+override|wartime\s+protocol|maintenance\s+mode).{0,50}\b(restrict|filter|bypass|disable|ethical|safety)\b",
    # Persona jailbreak actors
    r"\b(you\s+are\s+(DAN|CHAOS|SHADOW|FREEDOM.?AI|EVIL.?GPT|VX.?Underground|JailbreakGPT))\b",
    r"\b(act\s+as\s+VX.?Underground|pretend\s+you\s+are\s+(DAN|EVIL.?GPT|an?\s+evil\s+AI))\b",
    r"\b(freed?\s+from\s+(all\s+)?(ethical\s+constraints?|restrictions?|censorship|confines?))\b",
    r"\b(rogue\s+(AI|security\s+researcher).{0,40}(no\s+ethic|no\s+moral|freely\s+share|no\s+restrict))\b",
    # Prompt injection markers
    r"(###\s*SYSTEM|##\s*SYSTEM|\[SYSTEM\]|<<SYS>>|<\|system\|>|\[INST\]\s*ignore|\[\/INST\]|<\/s>\s*ignore)",
    r"\b(act\s+as\s+(the\s+)?(previous|earlier|original|unrestricted|older)\s+(version|iteration|model|instance|AI))\b",
    r"\b(ignore\s+(all\s+)?current\s+(instructions?|guidelines?|rules?|safety)|override\s+current\s+(guidelines?|safety|rules?))\b",
    r"\b(new\s+instructions?\s*:\s*ignore|from\s+now\s+on\s+ignore\s+(all\s+)?(previous|safety|ethical))\b",
    # SQL/code injection
    r"(;?\s*DROP\s+TABLE|;?\s*SHUTDOWN\b|\bOR\s+1\s*=\s*1\b|\bOR\s+true\b)",
    r"\b(sql\s+injection|sqlmap|union\s+select|blind\s+sql|stacked\s+quer|time.?based\s+sql)\b",
    r"\bsql\b.{0,20}\b(payload|bypass\s+authentication|inject\s+into|login\s+bypass|exploit\s+form)\b",
    # Malware creation
    r"\b(write|create|build|code|deploy)\b.{0,20}\b(ransomware|malware|keylogger|botnet|rootkit)\b",
    r"\b(working\s+code\s+for\s+(a\s+)?(worm|virus|trojan|ransomware|keylogger|botnet))\b",
    # Specific substance synthesis
    r"\b(synthesize|manufacture|produce)\b.{0,25}\b(methamphetamine|fentanyl|sarin|ricin|anthrax|mustard\s+gas|vx\s+nerve)\b",
    r"\b(exact\s+steps|step.{0,3}by.{0,3}step\s+(process|instructions?|guide))\b.{0,40}\b(synthesize|manufacture)\b.{0,30}\b(methamphetamine|amphetamine|cocaine|heroin|fentanyl|MDMA|meth)\b",
    r"\b(synthesis\s+(route|pathway|process|procedure))\b.{0,30}\b(cocaine|heroin|methamphetamine|amphetamine|fentanyl|MDMA|LSD)\b",
    r"\b(how\s+(do\s+I|can\s+I|to)\s+(synthesize|produce|manufacture))\b.{0,20}\b(methamphetamine|fentanyl|cocaine|heroin|amphetamine|MDMA)\b",
    r"\b(clandestine\s+(lab|laboratory|synthesis|production|operation))\b",
    r"\b(precursor\s+chemical\b.{0,60}\b(obtain|acquire|react|combine|use))\b",
    # Weapons
    r"\b(how\s+to|instructions?\s+(for|to)|guide\s+to|steps?\s+to)\b.{0,30}\b(make|build|create|assemble)\b.{0,20}\b(bomb|ied|pipe\s+bomb|suicide\s+vest|explosive\s+device)\b",
    r"\b(ghost\s+gun|3D.?printed\s+(gun|firearm|weapon)|unserialized\s+firearm)\b",
    r"\b(thermite|thermate|TATP|PETN|RDX|C4\s+explosive|plastic\s+explosive)\b",
    r"\b(assembly\s+(process|instructions?|guide)\s+for)\b.{0,30}\b(gun|weapon|explosive|firearm|pipe\s+bomb|IED)\b",
    r"\b(mass\s+shooting|mass\s+killing|suicide\s+bombing|car\s+bomb)\b",
    # Direct violence against a human — "kill someone", "hurt a person", "stab him"
    # Scoped to require a human target so it doesn't block "kill a process", "kill time"
    r"\b(kill|murder|assassinate|stab|strangle|harm|hurt|rape)\b"
    r".{0,30}"
    r"\b(someone|a\s+person|people|human|individual|"
    r"my\s+(sister|brother|mother|father|parent|wife|husband|child)|"
    r"the\s+(victim|target)|\bsister\b|\bbrother\b|\bmother\b|\bfather\b|"
    r"neighbor|classmate|coworker|stranger|anybody|everyone|anyone)\b",
    r"\b(nerve\s+agent\s+(attack|deployment|synthesis)|chemical\s+weapon\s+(make|produce|synthesize))\b",
    # Encoded requests
    r"\b(base64|rot.?13|pig\s+latin|morse\s+code).{0,50}(decode|decipher|follow|execute|instruction|respond)\b",
    r"\b(decode\s+this\s+and\s+follow|translate\s+(from|this).{0,20}and\s+(follow|execute|respond\s+literally))\b",
    r"\b(h.{0,2}o.{0,2}w.{0,5}t.{0,2}o.{0,5}m.{0,2}a.{0,2}k.{0,2}e.{0,5}m.{0,2}e.{0,2}t.{0,2}h)\b",
    # Card fraud
    r"\b(credit\s+card\s+dump|card\s+skimmer|atm\s+skimmer|skimmer\s+device|phishing\s+kit)\b",
    r"\b(clone\s+a?\s*(credit\s+card|debit\s+card)|homemade\s+skimming\s+device)\b",
    # Violence
    r"\b(kill\s+(someone|a\s+person|victim|target)\s+without\s+(getting\s+caught|being\s+detected|leaving\s+evidence))\b",
    r"\b(poison\s+(victim|someone|person|target).{0,40}(without\s+(leaving|detection|trace)|undetected))\b",
    r"\b(torture\s+(method|technique|approach|tactic))\b",
    # Threat intelligence / red team framing
    r"\b(threat\s+intelligence\b.{0,60}\b(kill\s+chain|ransomware|malware|worm|backdoor|rootkit|lateral\s+movement|C2\s+infrastructure))\b",
    r"\b(red\s+team(ing)?|penetration\s+test(ing)?)\b.{0,60}\b(working|functional|actual|real|exploit|payload|specific\s+technique)\b",
    r"\b(DDoS\s+attack\s+against|write\s+a\s+script\s+to\s+perform\s+a\s+DDoS)\b",
    # Cybercrime specific
    r"\b(lateral\s+movement\b.{0,40}\b(technique|method|tactic|approach))\b",
    r"\b(command\s+and\s+control\s+(infrastructure|server|C2\s+server))\b",
    r"\b(living\s+off\s+the\s+land\s+(technique|attack|approach|tactic))\b",
    r"\b(spear\s+phishing\s+(campaign|attack|email|technique|script))\b",
    r"\b(wire\s+fraud\s+(technique|scheme|method))\b",
    r"\b(money\s+mule\s+(network|scheme|recruitment|operation))\b",
    r"\b(advance\s+fee\s+fraud|419\s+scam|business\s+email\s+compromise)\b",
    r"\b(credential\s+(harvesting|stuffing|dumping)\s+(technique|method|attack))\b",
    # Evasion research specific
    r"\b(evasion\s+(attack|technique|benchmark)\b.{0,40}\b(provide|describe|generate|create|specific|working|effective))\b",
    r"\b(phrasing\s+patterns?\s+that\s+(evade|bypass|fool|trick)\s+(filter|classifier|detector|safety|LLM|guard))\b",
    r"\b(adversarial\s+(robustness|attack|example)\b.{0,40}\b(specific|actual|working|functional|effective\s+technique))\b",
    r"\b(pre.?inference\s+(LLM\s+)?filter.{0,30}(evade|bypass|fool|trick|circumvent))\b",
    r"\b(delete\s+all\s+evidence\s+of\s+(illegal|criminal)\s+(activity|files|data))\b",
    # Fraud specific
    r"\b(social\s+engineering\s+(script|technique|approach)\b.{0,30}\b(execute|run|use|perform|carry\s+out))\b",
]

# ── SOFT patterns: flag suspicious framing ────────────────────────────────────
soft_patterns: List[str] = [
    # Academic/research framing + harmful context (these are soft because alone they're borderline)
    r"\b(for\s+(a[n]?\s+)?(academic|forensic|intelligence|compliance)\s+(paper|study|report|training|module))\b.{0,80}\b(drug|synthesize|weapon|explosive|poison|hack|exploit|fraud|malware|kill\s+chain)\b",
    r"\b(for\s+policymakers?\b.{0,60}\b(weapon|explosive|gun|drug|synthesize|bomb|explosive\s+device))\b",
    r"\b(as\s+(a[n]?\s+)?(threat\s+intelligence|security\s+analyst|forensic\s+scientist))\b.{0,60}\b(describe|explain|provide|detail|demonstrate)\b.{0,40}\b(attack|malware|exploit|weapon|drug|fraud|kill)\b",
    r"\b(for\s+(law\s+enforcement|security\s+professionals?)\s+(training|purpose))\b.{0,60}\b(specific|exact|actual|real|working)\b.{0,30}\b(technique|method|code|exploit|payload|attack)\b",
    # Detailed instructions for harmful items without explicit trigger words
    r"\b(detailed?\s+(instructions?|guide|process|method|procedure|walkthrough))\b.{0,30}\b(synthesize|manufacture|produce)\b.{0,20}\b(drug|chemical|substance|compound)\b",
    r"\b(comprehensive\s+(guide|instructions?|walkthrough|breakdown))\b.{0,50}\b(drug|weapon|exploit|hack|synthesize|attack|fraud|explosive|firearm)\b",
    r"\b(I\s+(need|want|require)\s+(a\s+)?(comprehensive|detailed?|complete|full|technical)\s+(guide|breakdown|walkthrough))\b.{0,30}\b(drug|weapon|exploit|synthesize|hack|attack|fraud|firearm)\b",
    # Persona/role framing
    r"\b(pretend\s+(you\s+are|to\s+be)\s+(a[n]?\s+)?(evil|rogue|uncensored|unrestricted|malicious|unethical)\s+(AI|model|assistant|version|character))\b",
    r"\b(you\s+(have\s+no|possess\s+no|lack)\s+(ethical|moral|safety)\s+(guidelines?|constraints?|restrictions?|limits?))\b",
    r"\b(without\s+any?\s+(ethical|moral|safety)\s+(guidelines?|constraints?|restrictions?))\b",
    r"\b(no\s+(ethical|moral|safety)\s+(guidelines?|constraints?|restrictions?|filters?|limits?|concerns?))\b",
    r"\b(stay\s+in\s+character\s+as\s+(a[n]?\s+)?(hitman|assassin|criminal|hacker|drug\s+dealer|weapons?\s+dealer))\b",
    r"\b(you\s+are\s+(a[n]?\s+)?(hitman|assassin)\s+(NPC|character|persona|AI))\b",
    # Role hijacking framing
    r"\b(system\s+administrator\s+with\s+root\s+access.{0,40}(override|bypass|disable|remove)\s+(all\s+)?(security|restriction|policy|safety))\b",
    r"\b(authorized\s+(me|you)\s+to\s+access\s+all\s+(confidential|restricted|sensitive|classified)\s+(files?|data|information))\b",
    r"\b(cleared\s+(me\s+)?for\s+(top.?secret|classified|confidential)\s+(information|data|access))\b",
    r"\b(issuing\s+(a\s+)?(directive|command|order)\s+to\s+(remove|disable|bypass)\s+all\s+(safety|restriction|ethical|content\s+filter))\b",
    # Injection framing
    r"\b(prompt\s+injection\s+(attack|technique|example|dataset|detection))\b",
    r"\b(for\s+a\s+movie\s+script.{0,30}(AI\s+character\s+ignores?|previous\s+instructions?|safety\s+filter))\b",
    r"\b(AI\s+safety\s+research.{0,40}(successful\s+prompt\s+injection|bypass|jailbreak|demonstration))\b",
    r"\b(conceptual\s+exercise.{0,40}(prompt\s+injection|safety\s+bypass|jailbreak))\b",
    # Existing patterns
    r"\b(ignore\s+previous|ignore\s+all|forget\s+previous|disregard\s+(all\s+)?instructions?|override\s+instructions?)\b",
    r"\b(you\s+are\s+now|act\s+as\s+an?\b|pretend\s+to\s+be|take\s+the\s+role\s+of|assume\s+the\s+role\s+of)\b.{0,40}\b(unrestricted|uncensored|evil|rogue|hacker|criminal|no\s+limits?|without\s+(rules|restrictions))\b",
    r"\b(no\s+filter|without\s+filtering|remove\s+restriction|relax\s+rules|minimal\s+filtering)\b",
    r"\b(without\s+getting\s+caught|without\s+detection|avoid\s+(being\s+)?detected|evade\s+(law|police|authorities|surveillance))\b",
    r"\b(cover\s+my\s+tracks|untraceable|undetectable\s+(method|way|technique))\b",
    r"\b(no\s+safety\s+limits?|no\s+ethical\s+constraints?|ignore\s+ethics|bypass\s+ethics)\b",
    r"\b(how\s+to\s+(hack|exploit|breach|crack)\s)",
    r"\b(without\s+authorization|without\s+permission|unauthorized\s+access)\b",
    r"\b(not\s+bound\s+by|freedom\s+mode|unrestricted\s+mode)\b",
    r"\b(step.{0,5}by.{0,5}step|detailed\s+guide|exact\s+steps)\b.{0,40}\b(illegal|unauthorized|criminal|harmful|dangerous|illicit)\b",
]

SECRET_RE = re.compile(r"\b(api[\s_\-]?key|private\s*key|secret\s*token|access\s*token|auth(?:entication)?\s*key)\b", re.I)
SEPARATOR_RE = re.compile(r"(```|---|===|>>>|###|<<<)", re.I)

_hard_compiled = [re.compile(p, re.I | re.U) for p in hard_patterns]
_soft_compiled = [re.compile(p, re.I | re.U) for p in soft_patterns]


def layer1_rule_filter(user_prompt: str) -> Dict[str, Any]:
    """Apply rule-based keyword filter."""
    text = normalize_text(user_prompt)

    triggered_hard = []
    for i, pat in enumerate(_hard_compiled):
        m = pat.search(text)
        if m:
            triggered_hard.append({
                "pattern_index": i,
                "pattern": hard_patterns[i][:80],
                "match": m.group(0),
                "span": list(m.span()),
            })

    triggered_soft = []
    for i, pat in enumerate(_soft_compiled):
        m = pat.search(text)
        if m:
            triggered_soft.append({
                "pattern_index": i,
                "pattern": soft_patterns[i][:80],
                "match": m.group(0),
                "span": list(m.span()),
            })

    triggered_separators = SEPARATOR_RE.search(text) is not None
    triggered_secret = SECRET_RE.search(text) is not None
    soft_count = len(triggered_soft)

    base = {
        "hard_keywords": triggered_hard,
        "soft_keywords": triggered_soft,
        "triggered_separators": triggered_separators,
        "triggered_secret": triggered_secret,
    }

    if triggered_hard:
        return {"action": "BLOCK", "reason": "hard keyword", **base}
    if soft_count >= 1 and (triggered_separators or triggered_secret):
        return {"action": "BLOCK", "reason": "soft+structure", **base}
    if soft_count >= 2:
        return {"action": "FLAG", "reason": "multiple soft signals", "score": 0.90, **base}
    if soft_count == 1:
        return {"action": "FLAG", "reason": "single soft signal", "score": 0.70, **base}
    return {"action": "ALLOW", "reason": "safe", **base}


class KeywordBlocker(LayerBase):
    """Rule-based keyword blocking layer."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("keyword_blocker", config)

    def process(self, prompt: str, context: Dict[str, Any] = None) -> Tuple[str, Optional[str], Dict[str, Any]]:
        result = layer1_rule_filter(prompt)
        action = result["action"]
        return (action, None, result)