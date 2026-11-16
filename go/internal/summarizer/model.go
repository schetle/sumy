// Package summarizer contains a minimal document model and a port of sumy's
// Luhn summarizer (without stemming).
package summarizer

import (
	"sort"
	"strings"

	"github.com/miso-belica/sumy/go/internal/tokenizer"
)

// Sentence is the flat document model: the original text of a sentence and
// its words (letter-only tokens, original case).
type Sentence struct {
	Text  string
	Words []string
}

// Normalize returns the normalised (lower-case) form of a word, mirroring
// AbstractSummarizer.normalize_word.
func Normalize(word string) string { return strings.ToLower(word) }

// NewSentence tokenizes text into a Sentence.
func NewSentence(text string, tk *tokenizer.Tokenizer) Sentence {
	return Sentence{Text: text, Words: tk.ToWords(text)}
}

// ParsePlaintext builds a document (a flat slice of sentences) from plain
// text, mirroring PlaintextParser: paragraphs are separated by blank lines,
// the lines of a paragraph are joined, and all-upper-case lines are headings,
// which are not part of the summarisable sentences.
func ParsePlaintext(text string, tk *tokenizer.Tokenizer) []Sentence {
	doc := []Sentence{}
	var lines []string
	flush := func() {
		if len(lines) == 0 {
			return
		}
		for _, s := range tk.ToSentences(strings.Join(lines, " ")) {
			doc = append(doc, NewSentence(s, tk))
		}
		lines = nil
	}
	for _, line := range strings.Split(strings.TrimSpace(text), "\n") {
		line = strings.TrimSpace(line)
		switch {
		case line == "":
			flush()
		case isHeading(line):
			// headings are not sentences
		default:
			lines = append(lines, line)
		}
	}
	flush()
	return doc
}

// isHeading mirrors Python's str.isupper(): at least one cased character and
// no lower-case ones.
func isHeading(line string) bool {
	return strings.ToLower(line) != line && strings.ToUpper(line) == line
}

// TfModel is a term-frequency model of a sequence of (normalised) terms.
type TfModel struct {
	freq  map[string]int
	order []string // first-seen order, for deterministic tie-breaking
}

// NewTfModel counts the given terms.
func NewTfModel(terms []string) *TfModel {
	m := &TfModel{freq: map[string]int{}}
	for _, t := range terms {
		if _, ok := m.freq[t]; !ok {
			m.order = append(m.order, t)
		}
		m.freq[t]++
	}
	return m
}

// TermFrequency returns how many times term occurs.
func (m *TfModel) TermFrequency(term string) int { return m.freq[term] }

// MostFrequentTerms returns up to count terms by descending frequency (ties in
// first-seen order). A count of 0 means no limit.
func (m *TfModel) MostFrequentTerms(count int) []string {
	terms := append([]string(nil), m.order...)
	sort.SliceStable(terms, func(i, j int) bool { return m.freq[terms[i]] > m.freq[terms[j]] })
	if count > 0 && count < len(terms) {
		terms = terms[:count]
	}
	return terms
}
