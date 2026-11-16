package tokenizer

import (
	"reflect"
	"testing"
)

func mustNew(t *testing.T, lang string) *Tokenizer {
	t.Helper()
	tk, err := New(lang)
	if err != nil {
		t.Fatalf("New(%q): %v", lang, err)
	}
	return tk
}

// Ported from tests/test_tokenizers.py::test_missing_language.
func TestMissingLanguage(t *testing.T) {
	if _, err := New("klingon"); err == nil {
		t.Fatal("expected error for unsupported language")
	}
}

// Ported from test_language_getter and the slovak alias test: the original
// language name is kept even when the czech rules are used.
func TestLanguageGetter(t *testing.T) {
	for _, l := range []string{"english", "czech", "slovak"} {
		if got := mustNew(t, l).Language(); got != l {
			t.Errorf("Language() = %q, want %q", got, l)
		}
	}
}

// Ported from test_ensure_czech_tokenizer_available,
// test_tokenize_paragraph and test_slovak_alias_into_czech_tokenizer, plus
// extra cases for the rule-based splitter (design option 2).
//
// Deviations from Punkt are documented per case below.
func TestToSentences(t *testing.T) {
	tests := []struct {
		name, lang, in string
		want           []string
	}{
		{"czech", "czech", `
            Měl jsem sen, že toto je sen. Bylo to také zvláštní.
            Jakoby jsem plaval v moři rekurze.
        `, []string{
			"Měl jsem sen, že toto je sen.",
			"Bylo to také zvláštní.",
			"Jakoby jsem plaval v moři rekurze.",
		}},
		{"english paragraph", "english", `
            I am a very nice sentence with comma, but..
            This is next sentence. "I'm bored", said Pepek.
            Ou jee, duffman is here.
        `, []string{
			"I am a very nice sentence with comma, but..",
			"This is next sentence.",
			`"I'm bored", said Pepek.`,
			"Ou jee, duffman is here.",
		}},
		{"slovak alias", "slovak", `
            Je to veľmi fajn. Bodaj by nie.
            Ale na druhej strane čo je to oproti inému?
            To nechám na čitateľa.
        `, []string{
			"Je to veľmi fajn.",
			"Bodaj by nie.",
			"Ale na druhej strane čo je to oproti inému?",
			"To nechám na čitateľa.",
		}},
		{"empty", "english", "  \n ", []string{}},
		{"no terminator", "english", "just words", []string{"just words"}},
		// Closing quotes and brackets stay with the sentence they close.
		{"closing quote", "english", `He said "Stop!" Then he left. (Really.) Done.`,
			[]string{`He said "Stop!"`, "Then he left.", "(Really.)", "Done."}},
		{"question and exclamation", "english", "Really? Yes! Sure.",
			[]string{"Really?", "Yes!", "Sure."}},
		// Lower-case continuation never splits (matches the Luhn test text).
		{"ordinal numbers", "czech", "Měl problémy v 6. třídě. Nedopadl dobře.",
			[]string{"Měl problémy v 6. třídě.", "Nedopadl dobře."}},
		{"decimal", "english", "Pi is 3.14 roughly. Next one.",
			[]string{"Pi is 3.14 roughly.", "Next one."}},
		// Extra abbreviations from LANGUAGE_EXTRA_ABREVS.
		{"english extras", "english", "Use fruit, e.g. Apples are good. Smith et al. Wrote it. Also i.e. Pears.",
			[]string{"Use fruit, e.g. Apples are good.", "Smith et al. Wrote it.", "Also i.e. Pears."}},
		{"german extras", "german", "Das ist z. B. Hans gewesen. Vgl. Seite zwei. Es geht bzw. Nicht.",
			[]string{"Das ist z. B. Hans gewesen.", "Vgl. Seite zwei.", "Es geht bzw. Nicht."}},
		{"common abbreviation", "english", "Dr. Smith met Mr. Jones. They talked.",
			[]string{"Dr. Smith met Mr. Jones.", "They talked."}},
		// A word merely ending in an abbreviation is not an abbreviation.
		{"abbreviation word boundary", "german", "Das ist Zinc. Wir gehen.",
			[]string{"Das ist Zinc.", "Wir gehen."}},
		{"initials", "english", "J. K. Rowling wrote it. Good book.",
			[]string{"J. K. Rowling wrote it.", "Good book."}},
		// Punkt would likely split here; the rule-based splitter cannot tell an
		// initial from a one-letter word ending a sentence.
		{"initial at sentence end (differs from Punkt)", "english", "That is plan B. Next we go.",
			[]string{"That is plan B. Next we go."}},
		{"acronym", "english", "He lives in the U.S. Army bases are there.",
			[]string{"He lives in the U.S. Army bases are there."}},
		{"opening quote after boundary", "english", `It ended. "New start," she said.`,
			[]string{"It ended.", `"New start," she said.`}},
		{"digit start", "english", "First one. 2nd one.",
			[]string{"First one.", "2nd one."}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := mustNew(t, tc.lang).ToSentences(tc.in)
			if len(got) == 0 && len(tc.want) == 0 {
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got  %q\nwant %q", got, tc.want)
			}
		})
	}
}

// Ported from test_tokenize_sentence, plus letter-only filtering cases.
func TestToWords(t *testing.T) {
	tests := []struct {
		name, lang, in string
		want           []string
	}{
		{"ported", "english", "I am a very nice sentence with comma, but..",
			[]string{"I", "am", "a", "very", "nice", "sentence", "with", "comma", "but"}},
		{"digits and mixed dropped", "english", "In 2020 the 6th item x1 appeared",
			[]string{"In", "the", "item", "appeared"}},
		{"unicode letters", "czech", "Měl jsem sen, že toto je sen.",
			[]string{"Měl", "jsem", "sen", "že", "toto", "je", "sen"}},
		{"hyphenated dropped like NLTK", "english", "a well-known fact",
			[]string{"a", "fact"}},
		{"clitics split like Treebank", "english", "I'm sure they don't know it's here",
			[]string{"I", "sure", "they", "do", "know", "it", "here"}},
		{"punctuation only", "english", "... ?! --", []string{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := mustNew(t, tc.lang).ToWords(tc.in)
			if len(got) == 0 && len(tc.want) == 0 {
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got  %q\nwant %q", got, tc.want)
			}
		})
	}
}
