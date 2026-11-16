package summarizer

import (
	"math"
	"reflect"
	"testing"

	"github.com/miso-belica/sumy/go/internal/stopwords"
	"github.com/miso-belica/sumy/go/internal/tokenizer"
)

func czech(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tk, err := tokenizer.New("czech")
	if err != nil {
		t.Fatal(err)
	}
	return tk
}

func buildDoc(t *testing.T, sentences ...string) []Sentence {
	tk := czech(t)
	doc := []Sentence{}
	for _, s := range sentences {
		doc = append(doc, NewSentence(s, tk))
	}
	return doc
}

func luhn(stop ...string) *Luhn {
	l := NewLuhn()
	m := map[string]struct{}{}
	for _, w := range stop {
		m[w] = struct{}{}
	}
	l.SetStopWords(m)
	return l
}

func texts(ss []Sentence) []string {
	out := []string{}
	for _, s := range ss {
		out = append(out, s.Text)
	}
	return out
}

func TestEmptyDocument(t *testing.T) {
	if got := luhn().Summarize(nil, Count(10)); len(got) != 0 {
		t.Errorf("got %v", got)
	}
	if got := luhn().Summarize(nil, DefaultLength); len(got) != 0 {
		t.Errorf("got %v", got)
	}
}

func TestSingleSentence(t *testing.T) {
	doc := buildDoc(t, "Já jsem jedna věta")
	if got := luhn("já", "jsem").Summarize(doc, Count(10)); len(got) != 1 {
		t.Errorf("got %v", got)
	}
}

func TestTwoSentences(t *testing.T) {
	doc := buildDoc(t, "Já jsem 1. věta", "A já ta 2. vítězná výhra")
	got := texts(luhn("já", "jsem", "a", "ta").Summarize(doc, Count(10)))
	want := []string{"Já jsem 1. věta", "A já ta 2. vítězná výhra"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestTwoSentencesButOneWinner(t *testing.T) {
	doc := buildDoc(t, "Já jsem 1. vítězná ta věta", "A já ta 2. vítězná věta")
	got := texts(luhn("já", "jsem", "a", "ta").Summarize(doc, Count(1)))
	want := []string{"A já ta 2. vítězná věta"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestThreeSentences(t *testing.T) {
	a := "wa s s s wa s s s wa"
	b := "wb s wb s wb s s s s s s s s s wb"
	c := "wc s s wc s s wc"
	doc := buildDoc(t, a, b, c)
	l := luhn("s")
	tests := []struct {
		n    int
		want []string
	}{
		{1, []string{b}},
		{2, []string{b, c}},
		{3, []string{a, b, c}},
	}
	for _, tc := range tests {
		if got := texts(l.Summarize(doc, Count(tc.n))); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("n=%d got %q want %q", tc.n, got, tc.want)
		}
	}
}

func TestVariousWords(t *testing.T) {
	doc := buildDoc(t, "1 a", "2 b b", "3 c c c", "4 d d d", "5 z z z z", "6 e e e e e")
	l := luhn("1", "2", "3", "4", "5", "6")
	// Note: the digit tokens are not words (letter-only), so the stop words
	// are redundant here, as in the Python test where they are dropped by the
	// tokenizer too.
	tests := []struct {
		n    int
		want []string
	}{
		{1, []string{"6 e e e e e"}},
		{2, []string{"5 z z z z", "6 e e e e e"}},
		{3, []string{"3 c c c", "5 z z z z", "6 e e e e e"}},
	}
	for _, tc := range tests {
		if got := texts(l.Summarize(doc, Count(tc.n))); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("n=%d got %q want %q", tc.n, got, tc.want)
		}
	}
}

// Stable ordering of ties: equal ratings keep document order.
func TestTiesKeepDocumentOrder(t *testing.T) {
	doc := buildDoc(t, "x x", "y y", "x x", "y y")
	got := luhn().Summarize(doc, Count(2))
	if got[0].Text != "x x" || got[1].Text != "y y" {
		t.Errorf("got %q", texts(got))
	}
	// With the first two ranked (all rate 2), the first two sentences win.
	if len(got) != 2 {
		t.Fatalf("len %d", len(got))
	}
}

// Ported from test_real_example. Deviation: the Python test uses the Czech
// stemmer, which conflates "třídě"/"třídu" and "dětmi"/"dětí" so that the
// long fourth sentence also gets a positive rating. Without stemming only the
// first sentence has a rated chunk; the second slot is then a tie between
// zero-rated sentences, which the stable sort resolves in document order.
func TestRealExample(t *testing.T) {
	tk := czech(t)
	text := "Jednalo se o případ chlapce v 6. třídě, který měl problémy s učením. " +
		"Přerostly až v reparát z jazyka na konci školního roku. " +
		"Nedopadl bohužel dobře a tak musel opakovat 6. třídu, což se chlapci ani trochu nelíbilo. " +
		"Připadal si, že je mezi malými dětmi a realizoval se tím, že si ve třídě " +
		"o rok mladších dětí budoval vedoucí pozici. " +
		"Dost razantně. Fyzickou převahu měl, takže to nedalo až tak moc práce."
	doc := ParsePlaintext(text, tk)
	if len(doc) != 6 {
		t.Fatalf("expected 6 sentences, got %d: %q", len(doc), texts(doc))
	}
	l := NewLuhn()
	sw, err := stopwords.Get("czech")
	if err != nil {
		t.Fatal(err)
	}
	l.SetStopWords(sw)
	got := texts(l.Summarize(doc, Count(2)))
	want := []string{
		"Jednalo se o případ chlapce v 6. třídě, který měl problémy s učením.",
		"Přerostly až v reparát z jazyka na konci školního roku.",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %q\nwant %q", got, want)
	}
	if got := texts(l.Summarize(doc, Count(1))); !reflect.DeepEqual(got, want[:1]) {
		t.Errorf("top-1 got %q", got)
	}
}

func set(words ...string) map[string]struct{} {
	m := map[string]struct{}{}
	for _, w := range words {
		m[w] = struct{}{}
	}
	return m
}

func TestSignificantWordsPercentage(t *testing.T) {
	l := NewLuhn()
	l.SignificantPercentage = 1.0 / 5
	got := l.significantWords([]string{"wa", "wb", "wc", "wd", "we", "wf", "wg", "wh", "wi", "wj", "wa", "wb"})
	if !reflect.DeepEqual(got, set("wa", "wb")) {
		t.Errorf("got %v", got)
	}
}

func TestStopWordsNotSignificant(t *testing.T) {
	l := luhn("stop", "Halt", "SHUT", "HmMm")
	got := l.significantWords([]string{
		"stop", "Stop", "StOp", "STOP", "halt", "Halt", "HaLt", "HALT",
		"shut", "Shut", "ShUt", "SHUT", "hmmm", "Hmmm", "HmMm", "HMMM",
		"some", "relevant", "word", "some", "more", "relevant", "word",
	})
	if !reflect.DeepEqual(got, set("relevant", "some", "word")) {
		t.Errorf("got %v", got)
	}
}

func TestSentenceRating(t *testing.T) {
	text := "Nějaký muž šel kolem naší zahrady a žil pěkný život samotáře"
	tests := []struct {
		name string
		gap  int
		sig  []string
		sent string
		want float64
	}{
		{"zero rating", 0, nil, text, 0},
		{"single word", 0, []string{"muž"}, text, 0},
		{"single word before end", 0, []string{"život"}, text, 0},
		{"single word at end", 0, []string{"samotáře"}, text, 0},
		{"two chunks too far", 0, []string{"šel", "žil"}, text, 0},
		{"two chunks at begin", 0, []string{"muž", "šel"}, text, 2},
		{"two chunks before end", 0, []string{"pěkný", "život"}, text, 2},
		{"two chunks at end", 0, []string{"pěkný", "samotáře"}, text, 4.0 / 3},
		{"three chunks at begin", 0, []string{"nějaký", "muž", "šel"}, text, 3},
		{"three chunks at end", 0, []string{"pěkný", "život", "samotáře"}, text, 3},
		{"three chunks with gaps", 0, []string{"muž", "šel", "zahrady"}, text, 9.0 / 5},
		{"user gap", 6, []string{"muž", "šel", "pěkný"}, text, 9.0 / 8},
		{"1 gap", 0, []string{"w"}, "w s w s w", 9.0 / 5},
		{"2 gap", 0, []string{"w"}, "w s s w s s w", 9.0 / 7},
		{"3 gap", 0, []string{"w"}, "w s s s w s s s w", 1},
	}
	tk := czech(t)
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			l := NewLuhn()
			if tc.gap > 0 {
				l.MaxGapSize = tc.gap
			}
			got := l.RateSentence(NewSentence(tc.sent, tk), set(tc.sig...))
			if math.Abs(got-tc.want) > 1e-12 {
				t.Errorf("got %v want %v", got, tc.want)
			}
		})
	}
}

func TestLength(t *testing.T) {
	tests := []struct {
		in    string
		total int
		want  int
		err   bool
	}{
		{"3", 10, 3, false},
		{"30", 10, 10, false},
		{"20%", 10, 2, false},
		{"20%", 3, 1, false}, // at least one sentence
		{"0%", 5, 1, false},
		{"100%", 7, 7, false},
		{"x", 1, 0, true},
		{"-1", 1, 0, true},
		{"%", 1, 0, true},
		{"", 1, 0, true},
	}
	for _, tc := range tests {
		l, err := ParseLength(tc.in)
		if (err != nil) != tc.err {
			t.Errorf("ParseLength(%q) err = %v", tc.in, err)
			continue
		}
		if err == nil && l.Of(tc.total) != tc.want {
			t.Errorf("%q of %d = %d, want %d", tc.in, tc.total, l.Of(tc.total), tc.want)
		}
	}
	if DefaultLength.String() != "20%" {
		t.Errorf("default %s", DefaultLength)
	}
}

func TestParsePlaintext(t *testing.T) {
	text := "HEADING\n\nFirst sentence here. Second\nline sentence.\n\nThird one."
	tk, _ := tokenizer.New("english")
	doc := ParsePlaintext(text, tk)
	want := []string{"First sentence here.", "Second line sentence.", "Third one."}
	if got := texts(doc); !reflect.DeepEqual(got, want) {
		t.Errorf("got %q", got)
	}
	if !reflect.DeepEqual(doc[0].Words, []string{"First", "sentence", "here"}) {
		t.Errorf("words %v", doc[0].Words)
	}
	if len(ParsePlaintext("  \n", tk)) != 0 {
		t.Error("expected empty document")
	}
}

func TestTfModel(t *testing.T) {
	m := NewTfModel([]string{"a", "b", "b", "c", "c", "c"})
	if got := m.MostFrequentTerms(0); !reflect.DeepEqual(got, []string{"c", "b", "a"}) {
		t.Errorf("got %v", got)
	}
	if got := m.MostFrequentTerms(2); !reflect.DeepEqual(got, []string{"c", "b"}) {
		t.Errorf("got %v", got)
	}
	if m.TermFrequency("c") != 3 || m.TermFrequency("zzz") != 0 {
		t.Error("frequency")
	}
}
