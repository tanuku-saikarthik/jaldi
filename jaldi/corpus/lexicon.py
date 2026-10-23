"""Scenarios, action schemas and per-language surface forms.

Word order follows each language's natural grammar:
  English (SVO):  "I don't want to cancel my booking for tomorrow"   -> negation early
  Hindi   (SOV):  "मुझे कल की बुकिंग कैंसल नहीं करनी है"              -> negation after the verb root
  Hinglish (SOV): "mujhe kal ki booking cancel nahi karni hai"       -> same as Hindi

Every phenomenon is sampled at the same rate in every language, so any difference in late
meaning-flips comes from word order in the rendered sentences. That is a property of these
templates; whether real callers behave this way can only be measured on real speech.

Template syntax: "{slot}" is filled from the action; "[ ... ]" is kept only when every slot
inside it is present (used to drop a slot for hesitation turns).
"""

from __future__ import annotations

LANGS = ("en", "hi", "hinglish")

# ---------------------------------------------------------------- slot surface forms
_NUM = {
    2: ("two", "दो", "do"),
    3: ("three", "तीन", "teen"),
    4: ("four", "चार", "chaar"),
    5: ("five", "पांच", "paanch"),
    6: ("six", "छह", "chhah"),
    7: ("seven", "सात", "saat"),
    8: ("eight", "आठ", "aath"),
    9: ("nine", "नौ", "nau"),
    10: ("ten", "दस", "das"),
    11: ("eleven", "ग्यारह", "gyaarah"),
}


def _tri(en: str, hi: str, hl: str) -> dict:
    return {"en": en, "hi": hi, "hinglish": hl}


SLOT_VALUES: dict[str, dict[str, dict]] = {
    "party_size": {str(n): _tri(*_NUM[n]) for n in range(2, 9)},
    "time": {str(n): _tri(*_NUM[n]) for n in range(5, 12)},
    "day": {
        "today": _tri("today", "आज", "aaj"),
        "tomorrow": _tri("tomorrow", "कल", "kal"),
        "day_after": _tri("the day after tomorrow", "परसों", "parson"),
        "saturday": _tri("Saturday", "शनिवार", "Saturday"),
        "sunday": _tri("Sunday", "रविवार", "Sunday"),
        "monday": _tri("Monday", "सोमवार", "Monday"),
        "friday": _tri("Friday", "शुक्रवार", "Friday"),
    },
    "doctor": {
        "dentist": _tri("a dentist", "दांतों के डॉक्टर", "dentist"),
        "skin": _tri("a skin specialist", "स्किन स्पेशलिस्ट", "skin specialist"),
        "general": _tri("a general physician", "जनरल फिजिशियन", "general physician"),
    },
    "item": {
        "phone": _tri("phone", "फ़ोन", "phone"),
        "shoes": _tri("shoes", "जूतों", "shoes"),
        "headphones": _tri("headphones", "हेडफ़ोन", "headphones"),
        "books": _tri("books", "किताबों", "books"),
    },
    "city": {
        "delhi": _tri("Delhi", "दिल्ली", "Delhi"),
        "mumbai": _tri("Mumbai", "मुंबई", "Mumbai"),
        "chennai": _tri("Chennai", "चेन्नई", "Chennai"),
        "pune": _tri("Pune", "पुणे", "Pune"),
    },
    "place": {
        "airport": _tri("the airport", "एयरपोर्ट", "airport"),
        "station": _tri("the railway station", "रेलवे स्टेशन", "railway station"),
        "office": _tri("my office", "ऑफिस", "office"),
    },
    "mode": {
        "train": _tri("train", "ट्रेन", "train"),
        "bus": _tri("bus", "बस", "bus"),
        "flight": _tri("flight", "फ्लाइट", "flight"),
    },
}

# Afterthought form of a slot (used for hesitation turns and, behind a marker, for corrections)
AFTERTHOUGHT: dict[str, dict[str, str]] = {
    "party_size": _tri("for {v} people", "{v} लोगों के लिए", "{v} logon ke liye"),
    "time": _tri("at {v}", "{v} बजे", "{v} baje"),
    "day": _tri("for {v}", "{v} के लिए", "{v} ke liye"),
    "doctor": _tri("with {v}", "{v} के साथ", "{v} ke saath"),
    "item": _tri("the {v} one", "{v} वाला", "{v} wala"),
    "city": _tri("to {v}", "{v} के लिए", "{v} ke liye"),
    "place": _tri("to {v}", "{v} के लिए", "{v} ke liye"),
    "mode": _tri("by {v}", "{v} से", "{v} se"),
}

CORRECTION_MARKERS = {
    "en": ["no wait,", "sorry, I mean", "actually,"],
    "hi": ["नहीं,", "नहीं नहीं,", "मतलब"],
    "hinglish": ["nahi,", "nahi nahi,", "matlab"],
}

FILLERS = {
    "en": ["um,", "hi,", "so,", "okay so,"],
    "hi": ["जी,", "हाँ जी,", "अच्छा,", "सुनिए,"],
    "hinglish": ["haan ji,", "achha,", "so,", "suniye,"],
}

# Adjunct slots (not part of the object noun phrase). English appends them after the object;
# Hindi and Hinglish put them before the object.
ADJUNCT = {
    "en": {"day": "for {v}", "time": "at {v}"},
    "hi": {"day": "{v}", "time": "{v} बजे"},
    "hinglish": {"day": "{v}", "time": "{v} baje"},
}

# ---------------------------------------------------------------- scenarios and intents
# verb key -> surface per language (also the extractor's intent keywords)
VERBS = {
    "book": _tri("book", "बुक", "book"),
    "cancel": _tri("cancel", "कैंसल", "cancel"),
    "track": _tri("track", "ट्रैक", "track"),
    "reschedule": _tri("reschedule", "रीशेड्यूल", "reschedule"),
}

SCENARIOS: dict[str, dict] = {
    "restaurant": {
        "book_table": {
            "verb": "book",
            "required": ["party_size"],
            "optional": ["day", "time"],
            "negatable": True,
            "obj": _tri("a table[ for {party_size} people]", "[{party_size} लोगों के लिए ]टेबल", "[{party_size} logon ke liye ]table"),
            "gender": "f",
            "qualifier": {"slot": "seating", "value": "window", "text": _tri("near the window", "खिड़की के पास वाली", "window ke paas wali")},
        },
        "cancel_booking": {
            "verb": "cancel",
            "required": [],
            "optional": ["day"],
            "negatable": True,
            "obj": _tri("my booking[ for {day}]", "[{day} की ]बुकिंग", "[{day} ki ]booking"),
            "gender": "f",
        },
    },
    "clinic": {
        "book_appointment": {
            "verb": "book",
            "required": ["doctor"],
            "optional": ["day", "time"],
            "negatable": True,
            "obj": _tri("an appointment[ with {doctor}]", "[{doctor} का ]अपॉइंटमेंट", "[{doctor} ka ]appointment"),
            "gender": "m",
            "qualifier": {"slot": "consult", "value": "online", "text": _tri("as an online consultation", "ऑनलाइन वाला", "online wala")},
        },
        "cancel_appointment": {
            "verb": "cancel",
            "required": [],
            "optional": ["day"],
            "negatable": True,
            "obj": _tri("my appointment[ for {day}]", "[{day} का ]अपॉइंटमेंट", "[{day} ka ]appointment"),
            "gender": "m",
        },
    },
    "order": {
        "track_order": {
            "verb": "track",
            "required": ["item"],
            "optional": [],
            "negatable": False,
            "obj": _tri("my[ {item}] order", "अपना[ {item} वाला] ऑर्डर", "apna[ {item} wala] order"),
            "gender": "m",
        },
        "cancel_order": {
            "verb": "cancel",
            "required": ["item"],
            "optional": [],
            "negatable": True,
            "obj": _tri("my[ {item}] order", "अपना[ {item} वाला] ऑर्डर", "apna[ {item} wala] order"),
            "gender": "m",
            "qualifier": {"slot": "refund_to", "value": "wallet", "text": _tri("and refund it to my wallet", "और रिफंड वॉलेट में चाहिए", "aur refund wallet mein chahiye")},
        },
    },
    "delivery": {
        "reschedule_delivery": {
            "verb": "reschedule",
            "required": ["day"],
            "optional": [],
            "negatable": True,
            "obj": _tri("my delivery[ to {day}]", "अपनी डिलीवरी[ {day} के लिए]", "apni delivery[ {day} ke liye]"),
            "gender": "f",
            "qualifier": {"slot": "leave_at_door", "value": "yes", "text": _tri("and leave it at the door", "और दरवाज़े पर छोड़ दीजिए", "aur door pe chhod dena")},
        },
        "cancel_delivery": {
            "verb": "cancel",
            "required": [],
            "optional": [],
            "negatable": True,
            "obj": _tri("my delivery", "अपनी डिलीवरी", "apni delivery"),
            "gender": "f",
        },
    },
    "travel": {
        "book_ticket": {
            "verb": "book",
            "required": ["city"],
            "optional": ["mode", "day"],
            "negatable": True,
            "obj": _tri("a[ {mode}] ticket[ to {city}]", "[{city} के लिए ][{mode} का ]टिकट", "[{city} ke liye ][{mode} ka ]ticket"),
            "gender": "m",
            "qualifier": {"slot": "travel_class", "value": "ac", "text": _tri("in AC", "एसी में", "AC mein")},
        },
        "cancel_ticket": {
            "verb": "cancel",
            "required": [],
            "optional": ["city"],
            "negatable": True,
            "obj": _tri("my ticket[ to {city}]", "[{city} वाला ]टिकट", "[{city} wala ]ticket"),
            "gender": "m",
        },
    },
    "cab": {
        "book_cab": {
            "verb": "book",
            "required": ["place"],
            "optional": ["time"],
            "negatable": True,
            "obj": _tri("a cab[ to {place}]", "[{place} के लिए ]कैब", "[{place} ke liye ]cab"),
            "gender": "f",
            "qualifier": {"slot": "cab_type", "value": "suv", "text": _tri("an SUV please", "एसयूवी वाली", "SUV wali")},
        },
        "cancel_cab": {
            "verb": "cancel",
            "required": [],
            "optional": [],
            "negatable": True,
            "obj": _tri("my cab", "अपनी कैब", "apni cab"),
            "gender": "f",
        },
    },
}

# Sentence frames. Chunks are phrase-level units; pauses can only fall between chunks.
# "{A}" expands to the adjunct chunks, "{O}" object, "{V}" verb, "{K}" Hindi light-verb agreement.
FRAMES = {
    "en": {
        "pos": [
            ["I want to {V}", "{O}", "{A}"],
            ["I need to {V}", "{O}", "{A}"],
            ["Please {V}", "{O}", "{A}"],
            ["Can you {V}", "{O}", "{A}"],
            ["I'd like to {V}", "{O}", "{A}"],
        ],
        "neg": [
            ["I don't want to {V}", "{O}", "{A}"],
            ["Please don't {V}", "{O}", "{A}"],
        ],
    },
    "hi": {
        "pos": [
            ["मुझे", "{A}", "{O}", "{V} {K} है"],
            ["{A}", "{O}", "{V} कर दीजिए"],
            ["क्या आप", "{A}", "{O}", "{V} कर सकते हैं"],
        ],
        "neg": [
            ["मुझे", "{A}", "{O}", "{V}", "नहीं {K} है"],
            ["{A}", "{O}", "{V}", "मत कीजिए"],
        ],
    },
    "hinglish": {
        "pos": [
            ["mujhe", "{A}", "{O}", "{V} {K} hai"],
            ["{A}", "{O}", "{V} kar dijiye"],
            ["{A}", "{O}", "{V} kar do"],
            ["kya aap", "{A}", "{O}", "{V} kar sakte hain"],
        ],
        "neg": [
            ["mujhe", "{A}", "{O}", "{V}", "nahi {K} hai"],
            ["{A}", "{O}", "{V}", "mat kijiye"],
            ["{A}", "{O}", "{V}", "mat karna"],
        ],
    },
}

LIGHT_VERB = {"hi": {"f": "करनी", "m": "करना"}, "hinglish": {"f": "karni", "m": "karna"}}

# TTS speakers (Indic Parler-TTS model card; verify names before a GPU run).
# Test split uses held-out speakers only (speaker-disjoint).
SPEAKERS = {
    "en": {"train": ["Thoma", "Mary", "Swapna", "Dinesh"], "test": ["Kabir", "Sneha"]},
    "hi": {"train": ["Rohit", "Divya", "Aman"], "test": ["Rani"]},
    "hinglish": {"train": ["Rohit", "Divya", "Aman"], "test": ["Rani"]},
}

# Romanized Hindi -> Devanagari, for the mixed-script Hinglish variant (TTS script check).
# English-origin words stay in Latin script.
HINGLISH_TO_DEVANAGARI = {
    "mujhe": "मुझे", "kal": "कल", "aaj": "आज", "parson": "परसों", "ki": "की", "ka": "का",
    "ke": "के", "liye": "लिए", "logon": "लोगों", "apna": "अपना", "apni": "अपनी", "wala": "वाला",
    "wali": "वाली", "karni": "करनी", "karna": "करना", "hai": "है", "hain": "हैं", "nahi": "नहीं",
    "mat": "मत", "kijiye": "कीजिए", "kar": "कर", "dijiye": "दीजिए", "sakte": "सकते", "kya": "क्या",
    "aap": "आप", "baje": "बजे", "saath": "साथ", "se": "से", "mein": "में", "chahiye": "चाहिए",
    "aur": "और", "pe": "पे", "chhod": "छोड़", "dena": "देना", "paas": "पास", "haan": "हाँ",
    "ji": "जी", "achha": "अच्छा", "suniye": "सुनिए", "matlab": "मतलब", "do": "दो", "teen": "तीन",
    "chaar": "चार", "paanch": "पांच", "chhah": "छह", "saat": "सात", "aath": "आठ", "nau": "नौ",
    "das": "दस", "gyaarah": "ग्यारह",
}


def intent_spec(scenario: str, intent: str) -> dict:
    return SCENARIOS[scenario][intent]


def all_intents() -> list[tuple[str, str]]:
    return [(s, i) for s, ints in SCENARIOS.items() for i in ints]


def schema_slots(scenario: str, intent: str) -> set[str]:
    spec = intent_spec(scenario, intent)
    slots = set(spec["required"]) | set(spec["optional"])
    if "qualifier" in spec:
        slots.add(spec["qualifier"]["slot"])
    return slots


def validate_action(scenario: str, action: dict) -> list[str]:
    """Return a list of problems (empty if valid)."""
    errs = []
    if scenario not in SCENARIOS:
        return [f"unknown scenario {scenario}"]
    intent = action.get("intent")
    if intent not in SCENARIOS[scenario]:
        return [f"intent {intent} not in scenario {scenario}"]
    allowed = schema_slots(scenario, intent)
    spec = intent_spec(scenario, intent)
    for k, v in action.get("slots", {}).items():
        if k not in allowed:
            errs.append(f"slot {k} not allowed for {intent}")
        elif k in SLOT_VALUES and v not in SLOT_VALUES[k]:
            errs.append(f"bad value {v} for {k}")
        elif "qualifier" in spec and k == spec["qualifier"]["slot"] and v != spec["qualifier"]["value"]:
            errs.append(f"bad qualifier value {v}")
    if action.get("polarity") not in ("pos", "neg"):
        errs.append("polarity must be pos|neg")
    return errs
