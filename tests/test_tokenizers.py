import pytest

from sumy.nlp.tokenizers import Tokenizer


class TestTokenizer:
    def test_missing_language(self):
        with pytest.raises(LookupError):
            Tokenizer("klingon")

    def test_ensure_czech_tokenizer_available(self):
        tokenizer = Tokenizer("czech")
        assert "czech" == tokenizer.language

        sentences = tokenizer.to_sentences("""
            Měl jsem sen, že toto je sen. Bylo to také zvláštní.
            Jakoby jsem plaval v moři rekurze.
        """)

        expected = (
            "Měl jsem sen, že toto je sen.",
            "Bylo to také zvláštní.",
            "Jakoby jsem plaval v moři rekurze.",
        )
        assert expected == sentences

    def test_language_getter(self):
        tokenizer = Tokenizer("english")
        assert "english" == tokenizer.language

    def test_tokenize_sentence(self):
        tokenizer = Tokenizer("english")
        words = tokenizer.to_words("I am a very nice sentence with comma, but..")

        expected = (
            "I", "am", "a", "very", "nice", "sentence",
            "with", "comma", "but",
        )
        assert expected == words

    def test_tokenize_paragraph(self):
        tokenizer = Tokenizer("english")
        sentences = tokenizer.to_sentences("""
            I am a very nice sentence with comma, but..
            This is next sentence. "I'm bored", said Pepek.
            Ou jee, duffman is here.
        """)

        expected = (
            "I am a very nice sentence with comma, but..",
            "This is next sentence.",
            '"I\'m bored", said Pepek.',
            "Ou jee, duffman is here.",
        )
        assert expected == sentences

    def test_slovak_alias_into_czech_tokenizer(self):
        tokenizer = Tokenizer("slovak")
        assert tokenizer.language == "slovak"

        sentences = tokenizer.to_sentences("""
            Je to veľmi fajn. Bodaj by nie.
            Ale na druhej strane čo je to oproti inému?
            To nechám na čitateľa.
        """)

        expected = (
            "Je to veľmi fajn.",
            "Bodaj by nie.",
            "Ale na druhej strane čo je to oproti inému?",
            "To nechám na čitateľa.",
        )
        assert expected == sentences
