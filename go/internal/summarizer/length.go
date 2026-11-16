package summarizer

import (
	"fmt"
	"strconv"
	"strings"
)

// Length is how many sentences to select: an absolute count or a percentage
// of the document. It mirrors ItemsCount in sumy/utils.py.
type Length struct {
	n       int
	percent bool
}

// DefaultLength is 20% of the sentences, the default of the sumy CLI.
var DefaultLength = Length{n: 20, percent: true}

// Count returns an absolute sentence count.
func Count(n int) Length { return Length{n: n} }

// ParseLength parses "5" (count) or "20%" (percentage).
func ParseLength(s string) (Length, error) {
	s = strings.TrimSpace(s)
	pct := strings.HasSuffix(s, "%")
	num := strings.TrimSuffix(s, "%")
	n, err := strconv.Atoi(num)
	if err != nil || n < 0 {
		return Length{}, fmt.Errorf("invalid length %q: expected a non-negative integer or a percentage like 20%%", s)
	}
	return Length{n: n, percent: pct}, nil
}

// Of returns the number of items to take from a sequence of total items. A
// percentage always selects at least one item (as in Python), using integer
// arithmetic total*p/100.
func (l Length) Of(total int) int {
	n := l.n
	if l.percent {
		n = total * l.n / 100
		if n < 1 {
			n = 1
		}
	}
	if n > total {
		n = total
	}
	return n
}

func (l Length) String() string {
	if l.percent {
		return fmt.Sprintf("%d%%", l.n)
	}
	return strconv.Itoa(l.n)
}
