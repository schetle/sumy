package summarizer

import "sort"

// Luhn is a port of sumy.summarizers.luhn.LuhnSummarizer without stemming:
// words are compared in their normalised (lower-case) form.
type Luhn struct {
	// MaxGapSize is the number of consecutive non-significant words that ends
	// a chunk (default 4).
	MaxGapSize int
	// SignificantPercentage is the share of the (non-stop) words considered
	// when looking for significant ones (default 1, i.e. all).
	SignificantPercentage float64

	stopWords map[string]struct{}
}

// NewLuhn returns a Luhn summarizer with the default parameters.
func NewLuhn() *Luhn {
	return &Luhn{MaxGapSize: 4, SignificantPercentage: 1}
}

// SetStopWords sets the stop words; they are normalised like all words.
func (l *Luhn) SetStopWords(words map[string]struct{}) {
	l.stopWords = make(map[string]struct{}, len(words))
	for w := range words {
		l.stopWords[Normalize(w)] = struct{}{}
	}
}

// Summarize returns the best rated sentences in document order. count selects
// an absolute number or a percentage of sentences.
//
// Ratings are float64. Sentences are ranked by a stable sort on rating
// (descending), so ties keep document order, then the selection is re-sorted
// by document position.
func (l *Luhn) Summarize(doc []Sentence, count Length) []Sentence {
	var all []string
	for _, s := range doc {
		all = append(all, s.Words...)
	}
	significant := l.significantWords(all)

	type info struct {
		order  int
		rating float64
	}
	infos := make([]info, len(doc))
	for i, s := range doc {
		infos[i] = info{i, l.RateSentence(s, significant)}
	}
	sort.SliceStable(infos, func(a, b int) bool { return infos[a].rating > infos[b].rating })
	infos = infos[:count.Of(len(infos))]
	sort.Slice(infos, func(a, b int) bool { return infos[a].order < infos[b].order })

	out := make([]Sentence, len(infos))
	for i, in := range infos {
		out[i] = doc[in.order]
	}
	return out
}

// significantWords returns the normalised, non-stop words that occur more
// than once.
func (l *Luhn) significantWords(words []string) map[string]struct{} {
	terms := make([]string, 0, len(words))
	for _, w := range words {
		n := Normalize(w)
		if _, stop := l.stopWords[n]; !stop {
			terms = append(terms, n)
		}
	}
	model := NewTfModel(terms)
	best := int(float64(len(terms)) * l.SignificantPercentage)
	sig := map[string]struct{}{}
	for _, t := range model.MostFrequentTerms(best) {
		if model.TermFrequency(t) > 1 {
			sig[t] = struct{}{}
		}
	}
	return sig
}

// RateSentence returns the best chunk rating of the sentence (0 when it has
// no chunk with at least two significant words).
func (l *Luhn) RateSentence(s Sentence, significant map[string]struct{}) float64 {
	best := 0.0
	for _, c := range l.chunks(s, significant) {
		if r := chunkRating(c); r > best {
			best = r
		}
	}
	return best
}

// chunks splits the sentence into chunks of 0/1 flags (1 = significant word).
// A chunk starts at a significant word and ends after MaxGapSize consecutive
// non-significant words.
func (l *Luhn) chunks(s Sentence, significant map[string]struct{}) [][]int {
	var chunks [][]int
	inChunk := false
	for _, w := range s.Words {
		_, sig := significant[Normalize(w)]
		if sig && !inChunk {
			inChunk = true
			chunks = append(chunks, []int{1})
		} else if inChunk {
			v := 0
			if sig {
				v = 1
			}
			chunks[len(chunks)-1] = append(chunks[len(chunks)-1], v)
		}
		if len(chunks) > 0 && l.endsWithGap(chunks[len(chunks)-1]) {
			inChunk = false
		}
	}
	return chunks
}

func (l *Luhn) endsWithGap(chunk []int) bool {
	if len(chunk) < l.MaxGapSize {
		return false
	}
	for _, v := range chunk[len(chunk)-l.MaxGapSize:] {
		if v != 0 {
			return false
		}
	}
	return true
}

// chunkRating is significant^2 / length after removing trailing zeros; a
// chunk with a single significant word rates 0.
func chunkRating(chunk []int) float64 {
	end := len(chunk)
	for end > 0 && chunk[end-1] == 0 {
		end--
	}
	chunk = chunk[:end]
	sum := 0
	for _, v := range chunk {
		sum += v
	}
	if sum <= 1 {
		return 0
	}
	return float64(sum*sum) / float64(len(chunk))
}
