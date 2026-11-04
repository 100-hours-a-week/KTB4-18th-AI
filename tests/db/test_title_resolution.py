from types import SimpleNamespace

from db.search import pick_confident_title, resolve_title, title_distance


def rows(*titles):
    return [SimpleNamespace(title=title, artist='A', track_id=i) for i, title in enumerate(titles)]


def test_typo_with_clear_winner_is_chosen():
    assert pick_confident_title('The Day Bfore', rows('Today', 'The Day Before')).title == 'The Day Before'


def test_transposed_letters_count_as_one_edit():
    assert title_distance('Wsihlist', 'Wishlist') == 1


def test_replaced_word_is_a_different_title():
    assert pick_confident_title('When Love Passes Velvet', rows('When Love Passes By')) is None


def test_tied_candidates_are_not_chosen():
    assert pick_confident_title('Love Sung Now', rows('Love Song Now', 'Love Sang Now')) is None


def test_short_title_is_not_guessed():
    assert pick_confident_title('Folow', rows('Follow')) is None
    assert pick_confident_title('Shake', rows('Shame')) is None


def test_two_edits_need_a_long_title():
    assert pick_confident_title('This Cant Be 여름', rows('This Cant Be Us')) is None
    assert pick_confident_title('Through teh Nigth', rows('Through the Night')).title == 'Through the Night'


class Session:
    def __init__(self, listed, candidates):
        self.listed, self.candidates = listed, candidates

    def scalar(self, stmt):
        return 1 if self.listed else None

    def execute(self, stmt):
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: self.candidates))


def test_real_catalog_title_is_not_treated_as_typo():
    assert resolve_title(Session(True, rows('The Day Before')), 'The Day Befor', 'A') is None


def test_unlisted_title_is_resolved_within_artist():
    assert resolve_title(Session(False, rows('The Day Before')), 'The Day Bfore', 'A').title == 'The Day Before'


def test_case_and_punctuation_are_not_edits():
    assert title_distance('LOVE-DIVE', 'love dive') == 0


def test_duplicate_listing_of_one_song_is_not_ambiguous():
    assert pick_confident_title('Butter Flyy', rows('Butter Fly', 'butter fly')).title.lower() == 'butter fly'


def test_korean_typo_is_chosen():
    assert pick_confident_title('사랑은 늘 도망간지', rows('사랑은 늘 도망가지', '바다')).title == '사랑은 늘 도망가지'


def test_no_candidates_returns_none():
    assert pick_confident_title('Butter Flyy', []) is None
