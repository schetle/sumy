// Package tokenizer provides a rule-based sentence splitter and a word
// tokenizer. It mirrors sumy/nlp/tokenizers.py but does NOT aim for NLTK Punkt
// parity: Punkt is a trained, statistical model, while this is a small set of
// explicit rules (see ToSentences).
package tokenizer

import (
	"fmt"
	"strings"
	"unicode"
)

// languageAliases mirrors Tokenizer.LANGUAGE_ALIASES.
var languageAliases = map[string]string{
	"slovak": "czech",
}

// supported lists the languages for which NLTK ships a Punkt model; the Python
// Tokenizer raises LookupError for anything else, so New does the same.
var supported = map[string]bool{
	"czech": true, "danish": true, "dutch": true, "english": true,
	"estonian": true, "finnish": true, "french": true, "german": true,
	"greek": true, "italian": true, "norwegian": true, "polish": true,
	"portuguese": true, "russian": true, "slovene": true, "spanish": true,
	"swedish": true, "turkish": true,
}

// extraAbbreviations mirrors LANGUAGE_EXTRA_ABREVS (the final point is not
// included in the entries).
var extraAbbreviations = map[string][]string{
	"english": {"e.g", "al", "i.e"},
	"german":  {"al", "z.B", "Inc", "engl", "z. B", "vgl", "lat", "bzw", "S"},
}

// commonAbbreviations are additional frequent abbreviations that must not end
// a sentence, keyed by the (alias-resolved) tokenizer language. Punkt learns
// these from a corpus; here they are listed explicitly.
var commonAbbreviations = map[string][]string{
	"english": {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs",
		"gen", "col", "sgt", "capt", "lt", "cmdr", "rev", "hon", "inc", "ltd",
		"co", "corp", "no", "fig", "approx", "dept", "est"},
	"german": {"dr", "prof", "nr", "str", "ca", "usw", "evtl", "ggf", "sog", "bzw"},
	"czech": {"tj", "tzn", "tzv", "např", "mj", "apod", "resp", "atd", "mudr",
		"judr", "ing", "mgr", "bc", "prof", "doc", "sv", "str", "č", "tel"},
	"french": {"m", "mme", "mlle", "dr", "pr", "etc", "cf", "env", "av", "bd"},
}

// Tokenizer is a language dependent tokenizer of text documents.
type Tokenizer struct {
	language string
	abbrevs  [][]rune // lower-cased, without the final point
}

// New returns a Tokenizer for language ("slovak" is an alias of "czech"). An
// error is returned for unsupported languages.
func New(language string) (*Tokenizer, error) {
	tl := language
	if a, ok := languageAliases[language]; ok {
		tl = a
	}
	if !supported[tl] {
		return nil, fmt.Errorf("Tokenizer for language %q is not available.", language)
	}
	t := &Tokenizer{language: language}
	// Extra abbreviations are keyed by the original language name, as in Python.
	lists := [][]string{extraAbbreviations[language], commonAbbreviations[tl]}
	for _, l := range lists {
		for _, a := range l {
			t.abbrevs = append(t.abbrevs, lowerRunes(a))
		}
	}
	return t, nil
}

// Language returns the language the tokenizer was created with.
func (t *Tokenizer) Language() string { return t.language }

func lowerRunes(s string) []rune {
	rs := []rune(s)
	for i, r := range rs {
		rs[i] = unicode.ToLower(r)
	}
	return rs
}

func isTerminator(r rune) bool { return r == '.' || r == '!' || r == '?' || r == '…' }

func isCloser(r rune) bool {
	switch r {
	case '"', '\'', '”', '’', '»', ')', ']', '}', '›':
		return true
	}
	return false
}

func isOpener(r rune) bool {
	switch r {
	case '"', '\'', '“', '‘', '«', '(', '[', '{', '¿', '¡', '‹':
		return true
	}
	return false
}

// startsSentence reports whether r can begin a sentence: an uppercase letter,
// a digit or a letter of a script without case. Lower-case letters cannot.
func startsSentence(r rune) bool {
	return unicode.IsUpper(r) || unicode.IsTitle(r) || unicode.IsDigit(r) ||
		(unicode.IsLetter(r) && !unicode.IsLower(r))
}

// ToSentences splits a paragraph into trimmed sentences.
//
// Rules (design option 2 of the milestone):
//  1. A boundary is a run of terminators (. ! ? …), optionally followed by
//     closing quotes/brackets (so `said "Stop!" He` and `(Done.) Next` split
//     after the closer), followed by whitespace and a sentence start. The
//     start is an uppercase letter, digit or caseless letter, optionally
//     preceded by opening quotes/brackets. A lower-case start never splits
//     ("in 6. třídě").
//  2. A single "." is not a boundary after a known abbreviation (language
//     extras plus common ones, matched case-insensitively at a word start,
//     multi-word entries such as "z. B" included), after an initial (a single
//     upper-case letter: "J. K. Rowling") or after a dotted acronym such as
//     "U.S" or "e.g".
//
// Known deviations from Punkt: no corpus-learned abbreviations, an initial at
// a real sentence end ("Plan B. Next") is not split, and a boundary is never
// placed before a lower-case start.
func (t *Tokenizer) ToSentences(paragraph string) []string {
	rs := []rune(paragraph)
	n := len(rs)
	var out []string
	start := 0
	emit := func(end int) {
		s := strings.TrimSpace(string(rs[start:end]))
		if s != "" {
			out = append(out, s)
		}
	}
	for i := 0; i < n; {
		if !isTerminator(rs[i]) {
			i++
			continue
		}
		j := i
		for j < n && isTerminator(rs[j]) {
			j++
		}
		k := j
		for k < n && isCloser(rs[k]) {
			k++
		}
		if k < n && !unicode.IsSpace(rs[k]) {
			i = k
			continue
		}
		m := k
		for m < n && unicode.IsSpace(rs[m]) {
			m++
		}
		if m >= n { // end of text
			break
		}
		nx := m
		for nx < n && isOpener(rs[nx]) {
			nx++
		}
		if nx >= n || !startsSentence(rs[nx]) {
			i = k
			continue
		}
		if j-i == 1 && k == j && rs[i] == '.' && (t.endsWithAbbreviation(rs[:i]) || t.insideAbbreviation(rs, i)) {
			i = k
			continue
		}
		emit(k)
		start = k
		i = k
	}
	emit(n)
	return out
}

// endsWithAbbreviation reports whether pre (the text before a final ".")
// ends with an abbreviation, an initial or a dotted acronym.
func (t *Tokenizer) endsWithAbbreviation(pre []rune) bool {
	// Last whitespace-delimited token, without leading opening punctuation.
	b := len(pre)
	for b > 0 && !unicode.IsSpace(pre[b-1]) {
		b--
	}
	for b < len(pre) && isOpener(pre[b]) {
		b++
	}
	tok := pre[b:]
	if len(tok) == 0 {
		return false
	}
	if len(tok) == 1 && unicode.IsUpper(tok[0]) { // initial
		return true
	}
	if isDottedAcronym(tok) {
		return true
	}
	for _, a := range t.abbrevs {
		if len(a) > len(pre) {
			continue
		}
		off := len(pre) - len(a)
		match := true
		for x, r := range a {
			if unicode.ToLower(pre[off+x]) != r {
				match = false
				break
			}
		}
		if !match {
			continue
		}
		if off == 0 || !(unicode.IsLetter(pre[off-1]) || unicode.IsDigit(pre[off-1])) {
			return true
		}
	}
	return false
}

// insideAbbreviation reports whether the "." at index i is an inner point of
// an abbreviation containing a point (e.g. the first point of "z. B").
func (t *Tokenizer) insideAbbreviation(rs []rune, i int) bool {
	for _, a := range t.abbrevs {
		for p, r := range a {
			if r != '.' || p == len(a)-1 {
				continue
			}
			off := i - p
			if off < 0 || off+len(a) > len(rs) {
				continue
			}
			match := true
			for x, ar := range a {
				if unicode.ToLower(rs[off+x]) != ar {
					match = false
					break
				}
			}
			if match && (off == 0 || !(unicode.IsLetter(rs[off-1]) || unicode.IsDigit(rs[off-1]))) {
				return true
			}
		}
	}
	return false
}

// isDottedAcronym matches letter(.letter)+ such as "U.S" or "e.g".
func isDottedAcronym(tok []rune) bool {
	dots := 0
	for i, r := range tok {
		if i%2 == 0 {
			if !unicode.IsLetter(r) {
				return false
			}
		} else {
			if r != '.' {
				return false
			}
			dots++
		}
	}
	return dots > 0 && len(tok)%2 == 1
}

// ToWords splits a sentence into words, keeping only tokens made purely of
// letters (mirrors _WORD_PATTERN ^[^\W\d_]+$ applied to word_tokenize output).
//
// Tokens are runs of letters, digits and the connectors ' ’ - _, so that
// "well-known", "6th" and "x_1" stay single tokens that are then rejected, as
// with NLTK. English-style clitics are split off like the Treebank tokenizer
// does ("I'm" -> "I", "'m"; "don't" -> "do", "n't"); clitics are not letters
// only and so are dropped.
func (t *Tokenizer) ToWords(sentence string) []string {
	words := []string{}
	var cur []rune
	flush := func() {
		if len(cur) > 0 {
			words = appendWords(words, string(cur))
			cur = cur[:0]
		}
	}
	for _, r := range sentence {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || isConnector(r) {
			cur = append(cur, r)
		} else {
			flush()
		}
	}
	flush()
	return words
}

func isConnector(r rune) bool { return r == '\'' || r == '’' || r == '-' || r == '_' }

var clitics = []string{"n't", "'s", "'m", "'d", "'ll", "'re", "'ve"}

func appendWords(words []string, tok string) []string {
	tok = strings.Trim(tok, "'’-_")
	if tok == "" {
		return words
	}
	norm := strings.ReplaceAll(tok, "’", "'")
	lower := strings.ToLower(norm)
	for _, c := range clitics {
		if len(lower) > len(c) && strings.HasSuffix(lower, c) {
			tok = norm[:len(norm)-len(c)]
			break
		}
	}
	if isLetters(tok) {
		words = append(words, tok)
	}
	return words
}

func isLetters(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return true
}
