import pytest

from app.database import ChatDatabase


@pytest.fixture
def db(paths):
    d = ChatDatabase(paths.database)
    yield d
    d.close()


def _search(db, query):
    return [c.id for c in db.list_conversations(query)]


# (name, text stored, query) - written with \u escapes
CASES = [
    ("german", "\u00dcber uns", "\u00fcber"),
    ("french", "\u00c9l\u00e9phant au caf\u00e9", "\u00e9l\u00e9phant"),
    ("cyrillic", "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440", "\u043f\u0440\u0438\u0432\u0435\u0442"),
    ("greek", "\u0391\u03b8\u03ae\u03bd\u03b1", "\u03b1\u03b8\u03ae\u03bd\u03b1"),
]


@pytest.mark.parametrize("name,text,query", CASES)
def test_title_search_ignores_case(db, name, text, query):
    a = db.create_conversation(text)
    db.create_conversation("Other")
    for q in (query, query.upper(), query.lower(), query.title()):
        assert _search(db, q) == [a.id]


@pytest.mark.parametrize("name,text,query", CASES)
def test_message_search_ignores_case(db, name, text, query):
    a = db.create_conversation("Notes")
    db.add_message(a.id, "user", text)
    db.create_conversation("Other")
    for q in (query, query.upper(), query.lower(), query.title()):
        assert _search(db, q) == [a.id]


def test_sharp_s_matches_ss(db):
    a = db.create_conversation("Stra\u00dfe nach Hause")
    assert _search(db, "STRASSE") == [a.id]
    assert _search(db, "stra\u00dfe") == [a.id]


def test_wildcards_still_escaped_with_unicode(db):
    a = db.create_conversation("a_c \u00dcber")
    db.create_conversation("abc \u00dcber")
    c = db.create_conversation("C:\\temp 100% fertig")
    assert _search(db, "a_c") == [a.id]
    assert _search(db, "a%c") == []
    assert _search(db, "100%") == [c.id]
    assert _search(db, "C:\\TEMP") == [c.id]
