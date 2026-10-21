# -*- coding: utf8 -*-

import unittest

from sumy.nlp.tokenizers import Tokenizer


class TestTokenizer(unittest.TestCase):
    def test_missing_language(self):
        self.assertRaises(LookupError, Tokenizer, "klingon")

    def test_ensure_czech_tokenizer_available(self):
        tokenizer = Tokenizer("czech")
        self.assertEqual("czech", tokenizer.language)

        sentences = tokenizer.to_sentences("""
            Měl jsem sen, že toto je sen. Bylo to také zvláštní.
            Jakoby jsem plaval v moři rekurze.
        """)

        expected = (
            "Měl jsem sen, že toto je sen.",
            "Bylo to také zvláštní.",
            "Jakoby jsem plaval v moři rekurze.",
        )
        self.assertEqual(expected, sentences)

    def test_language_getter(self):
        tokenizer = Tokenizer("english")
        self.assertEqual("english", tokenizer.language)

    def test_tokenize_sentence(self):
        tokenizer = Tokenizer("english")
        words = tokenizer.to_words("I am a very nice sentence with comma, but..")

        # NOTE: modern NLTK's word tokenizer splits the trailing ".."
        # off of "but" instead of treating "but.." as a single token,
        # so "but" is correctly recognized as a word on its own.
        expected = (
            "I", "am", "a", "very", "nice", "sentence",
            "with", "comma", "but",
        )
        self.assertEqual(expected, words)

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
        self.assertEqual(expected, sentences)

    def test_sentence_boundary_detection_english(self):
        # Regression test for the punkt -> punkt_tab NLTK data migration:
        # common English abbreviations must not be mistaken for sentence
        # boundaries, while real sentence boundaries must still be found.
        tokenizer = Tokenizer("english")
        sentences = tokenizer.to_sentences(
            "Dr. Smith went to Washington, D.C. on Monday. "
            "He met Mr. Jones there. "
            "The U.S. economy grew by 3 percent. "
            "Mrs. Lee was pleased."
        )

        expected = (
            "Dr. Smith went to Washington, D.C. on Monday.",
            "He met Mr. Jones there.",
            "The U.S. economy grew by 3 percent.",
            "Mrs. Lee was pleased.",
        )
        self.assertEqual(expected, sentences)

    def test_sentence_boundary_detection_czech(self):
        # Same regression check as above but for a non-English punkt_tab
        # language model to make sure per-language data loads correctly.
        tokenizer = Tokenizer("czech")
        sentences = tokenizer.to_sentences(
            "Dr. Novák šel do práce. Bylo to v pondělí."
        )

        expected = (
            "Dr. Novák šel do práce.",
            "Bylo to v pondělí.",
        )
        self.assertEqual(expected, sentences)

    def test_slovak_alias_into_czech_tokenizer(self):
        tokenizer = Tokenizer("slovak")
        self.assertEqual(tokenizer.language, "slovak")

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
        self.assertEqual(expected, sentences)
