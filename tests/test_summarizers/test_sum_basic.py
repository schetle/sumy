from sumy.models.dom._sentence import Sentence
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.nlp.tokenizers import Tokenizer


EMPTY_STOP_WORDS = []
COMMON_STOP_WORDS = ["the", "and", "i"]


def _build_summarizer(stop_words):
    summarizer = SumBasicSummarizer()
    summarizer.stop_words = stop_words
    return summarizer


def test_empty_document(build_document):
    document = build_document()
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)

    returned = summarizer(document, 10)
    assert len(returned) == 0


def test_single_sentence(build_document):
    s = Sentence("I am one slightly longer sentence.", Tokenizer("english"))
    document = build_document([s])
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)

    returned = summarizer(document, 10)
    assert len(returned) == 1


def test_normalize_words():
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)
    sentence = "This iS A test 2 CHECk normalization."
    words_original = sentence.split()
    words_normalized = summarizer._normalize_words(words_original)

    words_correctly_normalized = "this is a test 2 check normalization.".split()
    assert words_normalized == words_correctly_normalized


def test_filter_out_stop_words():
    summarizer = _build_summarizer(COMMON_STOP_WORDS)
    sentence = "the dog and i went on a walk"
    words = sentence.split()
    words_filtered = summarizer._filter_out_stop_words(words)
    words_correctly_filtered = ["dog", "went", "on", "a", "walk"]
    assert words_filtered == words_correctly_filtered


def test_compute_word_freq():
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)

    words = ["one", "two", "three", "four"]
    freq = summarizer._compute_word_freq(words)
    assert freq.get("one", 0) == 1
    assert freq.get("two", 0) == 1
    assert freq.get("three", 0) == 1
    assert freq.get("four", 0) == 1

    words = ["one", "one", "two", "two"]
    freq = summarizer._compute_word_freq(words)
    assert freq.get("one", 0) == 2
    assert freq.get("two", 0) == 2
    assert freq.get("three", 0) == 0


def test_get_all_content_words_in_doc(build_document):
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)
    s0 = Sentence("One two three.", Tokenizer("english"))
    s1 = Sentence("One two three.", Tokenizer("english"))
    document = build_document([s0, s1])

    content_words = summarizer._get_all_content_words_in_doc(document.sentences)
    content_words_freq = {}
    for w in content_words:
        content_words_freq[w] = content_words_freq.get(w, 0) + 1
    content_words_correct = {"one": 2, "two": 2, "three": 2}
    assert content_words_freq == content_words_correct


def test_compute_tf(build_document):
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)
    s0 = Sentence("kicking soccer balls.", Tokenizer("english"))
    s1 = Sentence("eating chicken dumplings.", Tokenizer("english"))
    document = build_document([s0, s1])
    freq = summarizer._compute_tf(document.sentences)
    assert freq["kicking"] == 1/6
    assert freq["soccer"] == 1/6
    assert freq["balls"] == 1/6
    assert freq["eating"] == 1/6
    assert freq["chicken"] == 1/6
    assert freq["dumplings"] == 1/6

    document = build_document([s0, s0, s1])
    freq = summarizer._compute_tf(document.sentences)
    assert freq["kicking"] == 2/9
    assert freq["soccer"] == 2/9
    assert freq["balls"] == 2/9
    assert freq["eating"] == 1/9
    assert freq["chicken"] == 1/9
    assert freq["dumplings"] == 1/9


def test_compute_average_probability_of_words():
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)
    word_freq = {"one": 1/6, "two": 2/6, "three": 3/6}
    s0 = []
    s1 = ["one"]
    s2 = ["two", "three"]
    s3 = ["two", "three", "three"]
    EPS = 0.0001

    assert abs(summarizer._compute_average_probability_of_words(word_freq, s0) - 0) < EPS
    assert abs(summarizer._compute_average_probability_of_words(word_freq, s1) - 1/6) < EPS
    assert abs(summarizer._compute_average_probability_of_words(word_freq, s2) - 5/12) < EPS
    assert abs(summarizer._compute_average_probability_of_words(word_freq, s3) - 8/18) < EPS


def test_compute_ratings(build_document):
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)

    s0 = Sentence("Dog cat fish.", Tokenizer("english"))
    s1 = Sentence("Dog cat camel.", Tokenizer("english"))
    s2 = Sentence("Fish frog horse.", Tokenizer("english"))
    document = build_document([s0, s1, s2])

    ratings = summarizer._compute_ratings(document.sentences)
    assert ratings[s0] == 0
    assert ratings[s1] == -2
    assert ratings[s2] == -1

    # Due to the frequency discounting, after finding sentence s0,
    # s2 should come before s1 since only two of its words get discounted
    # rather than all 3 of s1's
    s0 = Sentence("one two three", Tokenizer("english"))
    s1 = Sentence("one two four", Tokenizer("english"))
    s2 = Sentence("three five six", Tokenizer("english"))
    document = build_document([s0, s1, s2])

    ratings = summarizer._compute_ratings(document.sentences)
    assert ratings[s0] == 0
    assert ratings[s1] == -2
    assert ratings[s2] == -1
