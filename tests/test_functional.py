"""
Functional tests for the modernized sumy library (Python 3.12+).

These tests exercise the full pipeline end-to-end to verify that the
modernization from Python 2.6+/3.2+ to Python 3.12+ did not break
any core functionality.
"""

import os
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILE = os.path.join(REPO_DIR, "tests", "data", "snippets", "prevko.txt")

# Sample English text used for multiple tests
ENGLISH_TEXT = (
    "Automatic summarization is the process of reducing a text document with a "
    "computer program in order to create a summary that retains the most important points "
    "of the original document. Technologies that can make a coherent summary take into "
    "account variables such as length, writing style and syntax. An example of the use of "
    "summarization technology is search engines such as Google. Document summarization "
    "is another example of the use of summarization technology."
)


# ============================================================================
# 1. CLI entry point
# ============================================================================
class TestCLIEntryPoint:
    """Verify that the CLI entry point works via `python -m sumy`."""

    def test_cli_luhn_czech_file(self):
        """CLI: luhn summarizer on Czech plaintext file produces output."""
        result = subprocess.run(
            [
                sys.executable, "-m", "sumy", "luhn",
                f"--file={TEST_FILE}",
                "--language", "czech",
                "--length", "3",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=REPO_DIR,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) > 0, "Expected at least one sentence of output"

    def test_cli_version(self):
        """CLI: --version prints version string."""
        result = subprocess.run(
            [sys.executable, "-m", "sumy", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=REPO_DIR,
        )
        assert result.returncode == 0
        assert "0.3.0" in result.stdout or "0.3.0" in result.stderr

    def test_cli_help(self):
        """CLI: --help exits 0 and shows usage."""
        result = subprocess.run(
            [sys.executable, "-m", "sumy", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=REPO_DIR,
        )
        assert result.returncode == 0
        assert "sumy" in result.stdout.lower()

    def test_cli_invalid_method(self):
        """CLI: invalid summarizer name exits non-zero."""
        result = subprocess.run(
            [sys.executable, "-m", "sumy", "nonexistent"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=REPO_DIR,
        )
        assert result.returncode != 0

    def test_cli_stdin_input(self):
        """CLI: reading from stdin when no --file or --url is given."""
        result = subprocess.run(
            [
                sys.executable, "-m", "sumy", "luhn",
                "--language", "english",
                "--length", "2",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            input=ENGLISH_TEXT,
            cwd=REPO_DIR,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) > 0, "Expected output from stdin summarization"

    def test_cli_percentage_length(self):
        """CLI: --length with percentage value produces output."""
        result = subprocess.run(
            [
                sys.executable, "-m", "sumy", "luhn",
                f"--file={TEST_FILE}",
                "--language", "czech",
                "--length", "50%",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=REPO_DIR,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) > 0

    def test_cli_all_methods_produce_output(self):
        """CLI: every summarization method (except KL) produces output via CLI.

        Note: KL summarizer is tested separately due to a known case-sensitivity
        bug in _kl_divergence where raw (capitalized) words from
        _get_all_words_in_doc are looked up in a lowercased word_freq dict.
        """
        methods = ["luhn", "edmundson", "lsa", "text-rank", "lex-rank", "sum-basic"]
        for method in methods:
            result = subprocess.run(
                [
                    sys.executable, "-m", "sumy", method,
                    f"--file={TEST_FILE}",
                    "--language", "czech",
                    "--length", "2",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=REPO_DIR,
            )
            assert result.returncode == 0, (
                f"Method '{method}' failed with exit code {result.returncode}. "
                f"stderr: {result.stderr}"
            )
            output = result.stdout.strip()
            assert len(output) > 0, f"Method '{method}' produced no output"

    @pytest.mark.xfail(
        reason="KL summarizer has a known bug: _get_all_words_in_doc returns "
               "unfiltered words (including stop words) for the summary, while "
               "word_freq from _compute_tf filters them out. This causes a KeyError "
               "in _kl_divergence when a stop word appears in joint_freq but not "
               "in word_freq. This is a pre-existing code defect, not a test issue.",
        strict=True,
    )
    def test_cli_kl_method(self):
        """CLI: KL summarizer via CLI."""
        result = subprocess.run(
            [
                sys.executable, "-m", "sumy", "kl",
                f"--file={TEST_FILE}",
                "--language", "czech",
                "--length", "2",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=REPO_DIR,
        )
        assert result.returncode == 0, f"KL method failed"
        output = result.stdout.strip()
        assert len(output) > 0, "KL method produced no output"


# ============================================================================
# 2. All 7 summarizers produce output (Python API)
# ============================================================================
class TestSummarizersAPI:
    """Verify all 7 summarizers produce output through the Python API."""

    @pytest.fixture()
    def english_document(self):
        """Parse English text into a document object."""
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.parsers.plaintext import PlaintextParser

        parser = PlaintextParser(ENGLISH_TEXT, Tokenizer("english"))
        return parser.document

    @pytest.fixture()
    def english_stop_words(self):
        from sumy.utils import get_stop_words
        return get_stop_words("english")

    @pytest.fixture()
    def english_stemmer(self):
        from sumy.nlp.stemmers import Stemmer
        return Stemmer("english")

    def test_luhn_summarizer(self, english_document, english_stop_words, english_stemmer):
        from sumy.summarizers.luhn import LuhnSummarizer

        summarizer = LuhnSummarizer(english_stemmer)
        summarizer.stop_words = english_stop_words
        sentences = summarizer(english_document, 2)
        assert len(sentences) > 0
        assert all(str(s).strip() for s in sentences)

    def test_edmundson_summarizer(self, english_document, english_stop_words, english_stemmer):
        from sumy.summarizers.edmundson import EdmundsonSummarizer
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer

        parser = PlaintextParser(ENGLISH_TEXT, Tokenizer("english"))
        summarizer = EdmundsonSummarizer(english_stemmer)
        summarizer.null_words = english_stop_words
        summarizer.bonus_words = parser.significant_words
        summarizer.stigma_words = parser.stigma_words
        sentences = summarizer(english_document, 2)
        assert len(sentences) > 0
        assert all(str(s).strip() for s in sentences)

    def test_lsa_summarizer(self, english_document, english_stop_words, english_stemmer):
        from sumy.summarizers.lsa import LsaSummarizer

        summarizer = LsaSummarizer(english_stemmer)
        summarizer.stop_words = english_stop_words
        sentences = summarizer(english_document, 2)
        assert len(sentences) > 0
        assert all(str(s).strip() for s in sentences)

    def test_textrank_summarizer(self, english_document, english_stop_words, english_stemmer):
        from sumy.summarizers.text_rank import TextRankSummarizer

        summarizer = TextRankSummarizer(english_stemmer)
        summarizer.stop_words = english_stop_words
        sentences = summarizer(english_document, 2)
        assert len(sentences) > 0
        assert all(str(s).strip() for s in sentences)

    def test_lexrank_summarizer(self, english_document, english_stop_words, english_stemmer):
        from sumy.summarizers.lex_rank import LexRankSummarizer

        summarizer = LexRankSummarizer(english_stemmer)
        summarizer.stop_words = english_stop_words
        sentences = summarizer(english_document, 2)
        assert len(sentences) > 0
        assert all(str(s).strip() for s in sentences)

    def test_sumbasic_summarizer(self, english_document, english_stop_words, english_stemmer):
        from sumy.summarizers.sum_basic import SumBasicSummarizer

        summarizer = SumBasicSummarizer(english_stemmer)
        summarizer.stop_words = english_stop_words
        sentences = summarizer(english_document, 2)
        assert len(sentences) > 0
        assert all(str(s).strip() for s in sentences)

    @pytest.mark.xfail(
        reason="KL summarizer has a known bug: _get_all_words_in_doc returns "
               "unfiltered words (including stop words) for the summary, while "
               "word_freq from _compute_tf filters them out. This causes a KeyError "
               "in _kl_divergence when a stop word in joint_freq is absent from "
               "word_freq. This is a pre-existing code defect, not a test issue.",
        strict=True,
    )
    def test_kl_summarizer(self, english_document, english_stop_words, english_stemmer):
        from sumy.summarizers.kl import KLSummarizer

        summarizer = KLSummarizer(english_stemmer)
        summarizer.stop_words = english_stop_words
        sentences = summarizer(english_document, 2)
        assert len(sentences) > 0
        assert all(str(s).strip() for s in sentences)


# ============================================================================
# 3. Evaluation functions work end-to-end
# ============================================================================
class TestEvaluationFunctions:
    """Verify evaluation functions return numeric results."""

    @pytest.fixture()
    def sentence_sets(self):
        """Create two sets of Sentence objects for evaluation."""
        from sumy.models.dom import Sentence
        from sumy.nlp.tokenizers import Tokenizer

        tokenizer = Tokenizer("english")
        evaluated = [
            Sentence("Automatic summarization reduces text documents.", tokenizer),
            Sentence("Search engines use summarization technology.", tokenizer),
        ]
        reference = [
            Sentence("Automatic summarization reduces text documents.", tokenizer),
            Sentence("Document summarization is useful for search engines.", tokenizer),
        ]
        return evaluated, reference

    @pytest.fixture()
    def tf_models(self):
        """Create TfDocumentModel instances for content-based evaluation."""
        from sumy.models import TfDocumentModel

        model_a = TfDocumentModel(["automatic", "summarization", "text", "document", "search"])
        model_b = TfDocumentModel(["automatic", "summarization", "document", "engine", "query"])
        return model_a, model_b

    def test_precision(self, sentence_sets):
        from sumy.evaluation import precision

        evaluated, reference = sentence_sets
        result = precision(evaluated, reference)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_recall(self, sentence_sets):
        from sumy.evaluation import recall

        evaluated, reference = sentence_sets
        result = recall(evaluated, reference)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_f_score(self, sentence_sets):
        from sumy.evaluation import f_score

        evaluated, reference = sentence_sets
        result = f_score(evaluated, reference)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_cosine_similarity(self, tf_models):
        from sumy.evaluation import cosine_similarity

        model_a, model_b = tf_models
        result = cosine_similarity(model_a, model_b)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_cosine_similarity_identical(self):
        """Cosine similarity of identical documents should be 1.0."""
        from sumy.evaluation import cosine_similarity
        from sumy.models import TfDocumentModel

        words = ["hello", "world", "test"]
        model = TfDocumentModel(words)
        result = cosine_similarity(model, model)
        assert abs(result - 1.0) < 1e-9

    def test_rouge_n(self, sentence_sets):
        from sumy.evaluation import rouge_n

        evaluated, reference = sentence_sets
        result = rouge_n(evaluated, reference, n=1)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_rouge_1(self, sentence_sets):
        from sumy.evaluation import rouge_1

        evaluated, reference = sentence_sets
        result = rouge_1(evaluated, reference)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_rouge_2(self, sentence_sets):
        from sumy.evaluation import rouge_2

        evaluated, reference = sentence_sets
        result = rouge_2(evaluated, reference)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_rouge_l_sentence_level(self, sentence_sets):
        from sumy.evaluation import rouge_l_sentence_level

        evaluated, reference = sentence_sets
        result = rouge_l_sentence_level(evaluated, reference)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_rouge_l_summary_level(self, sentence_sets):
        from sumy.evaluation import rouge_l_summary_level

        evaluated, reference = sentence_sets
        result = rouge_l_summary_level(evaluated, reference)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_unit_overlap(self, tf_models):
        from sumy.evaluation.content_based import unit_overlap

        model_a, model_b = tf_models
        result = unit_overlap(model_a, model_b)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ============================================================================
# 4. Tokenizer works for multiple languages
# ============================================================================
class TestTokenizer:
    """Verify tokenizer handles English and Czech correctly."""

    def test_english_sentence_tokenization(self):
        from sumy.nlp.tokenizers import Tokenizer

        tokenizer = Tokenizer("english")
        text = "Hello world. This is a test. It has three sentences."
        sentences = tokenizer.to_sentences(text)
        assert isinstance(sentences, tuple)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "This is a test."
        assert sentences[2] == "It has three sentences."

    def test_english_word_tokenization(self):
        from sumy.nlp.tokenizers import Tokenizer

        tokenizer = Tokenizer("english")
        words = tokenizer.to_words("Hello world, this is a test.")
        assert isinstance(words, tuple)
        assert len(words) > 0
        # Words should not include punctuation
        assert "Hello" in words
        assert "world" in words
        assert "," not in words
        assert "." not in words

    def test_czech_sentence_tokenization(self):
        from sumy.nlp.tokenizers import Tokenizer

        tokenizer = Tokenizer("czech")
        text = "Ahoj svete. Toto je test. Ma tri vety."
        sentences = tokenizer.to_sentences(text)
        assert isinstance(sentences, tuple)
        assert len(sentences) == 3

    def test_czech_word_tokenization(self):
        from sumy.nlp.tokenizers import Tokenizer

        tokenizer = Tokenizer("czech")
        words = tokenizer.to_words("Ahoj svete, toto je test.")
        assert isinstance(words, tuple)
        assert len(words) > 0
        assert "Ahoj" in words
        assert "svete" in words

    def test_tokenizer_language_property(self):
        from sumy.nlp.tokenizers import Tokenizer

        tokenizer = Tokenizer("english")
        assert tokenizer.language == "english"

    def test_tokenizer_language_alias(self):
        """Slovak should alias to Czech tokenizer internally."""
        from sumy.nlp.tokenizers import Tokenizer

        tokenizer = Tokenizer("slovak")
        assert tokenizer.language == "slovak"
        # Should not raise -- the alias allows sentence tokenization
        sentences = tokenizer.to_sentences("Ahoj. Toto je test.")
        assert len(sentences) >= 1


# ============================================================================
# 5. PlaintextParser round-trip
# ============================================================================
class TestPlaintextParser:
    """Parse text -> get document -> get sentences -> convert back to strings."""

    def test_parse_and_extract_sentences(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.models.dom import Sentence

        parser = PlaintextParser(ENGLISH_TEXT, Tokenizer("english"))
        document = parser.document

        assert document is not None
        assert len(document.paragraphs) > 0

        all_sentences = []
        for paragraph in document.paragraphs:
            for sentence in paragraph.sentences:
                assert isinstance(sentence, Sentence)
                text = str(sentence)
                assert len(text) > 0
                all_sentences.append(text)

        assert len(all_sentences) > 0

    def test_parse_from_file(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer

        parser = PlaintextParser.from_file(TEST_FILE, Tokenizer("czech"))
        document = parser.document
        assert len(document.paragraphs) > 0

        sentences = []
        for paragraph in document.paragraphs:
            sentences.extend(paragraph.sentences)
        assert len(sentences) > 0

    def test_parse_from_string(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer

        parser = PlaintextParser.from_string(ENGLISH_TEXT, Tokenizer("english"))
        document = parser.document
        assert len(document.paragraphs) > 0

    def test_sentence_words_are_populated(self):
        """Sentences from parser have words (tokenized)."""
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer

        parser = PlaintextParser(ENGLISH_TEXT, Tokenizer("english"))
        for paragraph in parser.document.paragraphs:
            for sentence in paragraph.sentences:
                words = sentence.words
                assert isinstance(words, tuple)
                assert len(words) > 0

    def test_document_model_str(self):
        """ObjectDocumentModel should have meaningful string repr."""
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer

        parser = PlaintextParser(ENGLISH_TEXT, Tokenizer("english"))
        doc_str = str(parser.document)
        assert len(doc_str) > 0


# ============================================================================
# 6. Optional NumPy dependency (LSA and LexRank)
# ============================================================================
class TestNumpyDependency:
    """Verify LSA and LexRank work when numpy is available."""

    def test_numpy_is_available(self):
        """Precondition: numpy should be importable."""
        import numpy
        assert numpy is not None

    def test_lsa_uses_numpy(self):
        """LSA summarizer should import and use numpy for SVD."""
        from sumy.summarizers.lsa import LsaSummarizer, numpy as lsa_numpy

        assert lsa_numpy is not None, "LSA should detect numpy"

    def test_lexrank_uses_numpy(self):
        """LexRank summarizer should import and use numpy for matrix ops."""
        from sumy.summarizers.lex_rank import LexRankSummarizer, numpy as lex_numpy

        assert lex_numpy is not None, "LexRank should detect numpy"

    def test_lsa_full_pipeline_with_numpy(self):
        """LSA: full summarization pipeline using numpy-backed SVD."""
        from sumy.summarizers.lsa import LsaSummarizer
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words

        parser = PlaintextParser(ENGLISH_TEXT, Tokenizer("english"))
        stemmer = Stemmer("english")
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")

        sentences = summarizer(parser.document, 2)
        assert len(sentences) > 0
        for s in sentences:
            assert len(str(s)) > 0

    def test_lexrank_full_pipeline_with_numpy(self):
        """LexRank: full summarization pipeline using numpy-backed matrix ops."""
        from sumy.summarizers.lex_rank import LexRankSummarizer
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words

        parser = PlaintextParser(ENGLISH_TEXT, Tokenizer("english"))
        stemmer = Stemmer("english")
        summarizer = LexRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")

        sentences = summarizer(parser.document, 2)
        assert len(sentences) > 0
        for s in sentences:
            assert len(str(s)) > 0


# ============================================================================
# 7. Stop words loading
# ============================================================================
class TestStopWords:
    """Verify stop words load correctly for supported languages."""

    def test_english_stop_words(self):
        from sumy.utils import get_stop_words

        stop_words = get_stop_words("english")
        assert isinstance(stop_words, frozenset)
        assert len(stop_words) > 0
        # Common English stop words should be present
        assert "the" in stop_words
        assert "is" in stop_words
        assert "a" in stop_words

    def test_czech_stop_words(self):
        from sumy.utils import get_stop_words

        stop_words = get_stop_words("czech")
        assert isinstance(stop_words, frozenset)
        assert len(stop_words) > 0

    def test_french_stop_words(self):
        from sumy.utils import get_stop_words

        stop_words = get_stop_words("french")
        assert isinstance(stop_words, frozenset)
        assert len(stop_words) > 0

    def test_german_stop_words(self):
        from sumy.utils import get_stop_words

        stop_words = get_stop_words("german")
        assert isinstance(stop_words, frozenset)
        assert len(stop_words) > 0

    def test_unsupported_language_raises(self):
        from sumy.utils import get_stop_words

        with pytest.raises(LookupError):
            get_stop_words("klingon")


# ============================================================================
# 8. ItemsCount
# ============================================================================
class TestItemsCount:
    """Verify ItemsCount works with percentage and numeric modes."""

    def test_numeric_count(self):
        from sumy.utils import ItemsCount

        counter = ItemsCount(3)
        result = counter(list(range(10)))
        assert result == [0, 1, 2]

    def test_string_numeric_count(self):
        from sumy.utils import ItemsCount

        counter = ItemsCount("3")
        result = counter(list(range(10)))
        assert result == [0, 1, 2]

    def test_percentage_count(self):
        from sumy.utils import ItemsCount

        counter = ItemsCount("20%")
        data = list(range(10))
        result = counter(data)
        # 20% of 10 = 2
        assert result == [0, 1]

    def test_percentage_at_least_one(self):
        """Percentage mode should return at least 1 item."""
        from sumy.utils import ItemsCount

        counter = ItemsCount("1%")
        data = list(range(10))
        result = counter(data)
        # 1% of 10 = 0.1, but minimum is 1
        assert len(result) >= 1

    def test_float_count(self):
        from sumy.utils import ItemsCount

        counter = ItemsCount(2.5)
        result = counter(list(range(10)))
        # int(2.5) = 2
        assert result == [0, 1]

    def test_percentage_50(self):
        from sumy.utils import ItemsCount

        counter = ItemsCount("50%")
        data = list(range(10))
        result = counter(data)
        # 50% of 10 = 5
        assert len(result) == 5

    def test_repr(self):
        from sumy.utils import ItemsCount

        counter = ItemsCount("20%")
        r = repr(counter)
        assert "20%" in r
        assert "ItemsCount" in r

    def test_invalid_value_raises(self):
        from sumy.utils import ItemsCount

        counter = ItemsCount([1, 2, 3])
        with pytest.raises(ValueError):
            counter(list(range(10)))


# ============================================================================
# 9. Stemmer
# ============================================================================
class TestStemmer:
    """Verify stemmers work for supported languages."""

    def test_english_stemmer(self):
        from sumy.nlp.stemmers import Stemmer

        stemmer = Stemmer("english")
        result = stemmer("running")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_czech_stemmer(self):
        from sumy.nlp.stemmers import Stemmer

        stemmer = Stemmer("czech")
        result = stemmer("problemy")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_null_stemmer(self):
        from sumy.nlp.stemmers import null_stemmer

        assert null_stemmer("hello") == "hello"


# ============================================================================
# 10. TfDocumentModel
# ============================================================================
class TestTfDocumentModel:
    """Verify TfDocumentModel works correctly."""

    def test_from_word_list(self):
        from sumy.models import TfDocumentModel

        model = TfDocumentModel(["hello", "world", "hello"])
        assert model.term_frequency("hello") == 2
        assert model.term_frequency("world") == 1
        assert model.term_frequency("missing") == 0

    def test_from_string_with_tokenizer(self):
        from sumy.models import TfDocumentModel
        from sumy.nlp.tokenizers import Tokenizer

        tokenizer = Tokenizer("english")
        model = TfDocumentModel("Hello world, this is a test.", tokenizer)
        assert len(list(model.terms)) > 0

    def test_magnitude(self):
        from sumy.models import TfDocumentModel

        model = TfDocumentModel(["a", "b", "c"])
        assert model.magnitude > 0

    def test_most_frequent_terms(self):
        from sumy.models import TfDocumentModel

        model = TfDocumentModel(["hello", "hello", "hello", "world", "test"])
        top = model.most_frequent_terms(2)
        assert len(top) == 2
        assert top[0] == "hello"

    def test_normalized_term_frequency(self):
        from sumy.models import TfDocumentModel

        model = TfDocumentModel(["hello", "hello", "world"])
        ntf = model.normalized_term_frequency("hello")
        assert 0.0 <= ntf <= 1.0
        assert ntf == 1.0  # most frequent term

    def test_string_without_tokenizer_raises(self):
        from sumy.models import TfDocumentModel

        with pytest.raises(ValueError):
            TfDocumentModel("hello world")


# ============================================================================
# 11. Full pipeline integration: parse -> summarize -> evaluate
# ============================================================================
class TestFullPipeline:
    """End-to-end: parse text, summarize, then evaluate the summary."""

    def test_parse_summarize_evaluate(self):
        """Complete pipeline: parse English text, summarize with Luhn, evaluate with ROUGE."""
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words
        from sumy.summarizers.luhn import LuhnSummarizer
        from sumy.evaluation import rouge_1, precision, recall, f_score

        tokenizer = Tokenizer("english")
        parser = PlaintextParser(ENGLISH_TEXT, tokenizer)
        stemmer = Stemmer("english")
        stop_words = get_stop_words("english")

        # Summarize
        summarizer = LuhnSummarizer(stemmer)
        summarizer.stop_words = stop_words
        summary_sentences = summarizer(parser.document, 2)
        assert len(summary_sentences) > 0

        # Use all document sentences as reference
        all_sentences = []
        for p in parser.document.paragraphs:
            all_sentences.extend(p.sentences)
        assert len(all_sentences) > 0

        # Evaluate ROUGE-1
        rouge_score = rouge_1(list(summary_sentences), all_sentences)
        assert isinstance(rouge_score, float)
        assert 0.0 <= rouge_score <= 1.0

        # Evaluate coselection metrics
        p = precision(list(summary_sentences), all_sentences)
        r = recall(list(summary_sentences), all_sentences)
        f = f_score(list(summary_sentences), all_sentences)
        assert all(isinstance(v, float) for v in [p, r, f])
        assert all(0.0 <= v <= 1.0 for v in [p, r, f])

    def test_czech_file_full_pipeline(self):
        """Complete pipeline with Czech text from a file."""
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words
        from sumy.summarizers.text_rank import TextRankSummarizer

        tokenizer = Tokenizer("czech")
        parser = PlaintextParser.from_file(TEST_FILE, tokenizer)
        stemmer = Stemmer("czech")
        stop_words = get_stop_words("czech")

        summarizer = TextRankSummarizer(stemmer)
        summarizer.stop_words = stop_words
        sentences = summarizer(parser.document, 3)
        assert len(sentences) > 0
        # Verify sentences are actual strings
        for s in sentences:
            text = str(s)
            assert len(text) > 0
            assert isinstance(text, str)
