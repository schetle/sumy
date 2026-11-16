// Package stopwords provides embedded stop-word lists and a loader for
// user-supplied files. It mirrors get_stop_words/read_stop_words in
// sumy/utils.py.
package stopwords

import (
	"bufio"
	"embed"
	"fmt"
	"io"
	"os"
	"strings"
)

//go:embed data/*.txt
var data embed.FS

// Set is a set of stop words.
type Set map[string]struct{}

// Contains reports whether w is in the set.
func (s Set) Contains(w string) bool {
	_, ok := s[w]
	return ok
}

// Get returns the embedded stop words for language (e.g. "english").
// An error is returned when no list is available for the language.
func Get(language string) (Set, error) {
	// Reject anything that could be a path rather than a language name.
	if language == "" || strings.ContainsAny(language, `/\.`) {
		return nil, fmt.Errorf("Stop-words are not available for language %s.", language)
	}
	f, err := data.Open("data/" + language + ".txt")
	if err != nil {
		return nil, fmt.Errorf("Stop-words are not available for language %s.", language)
	}
	defer f.Close()
	return read(f)
}

// ReadFile reads a UTF-8 file with one stop word per line.
func ReadFile(path string) (Set, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return read(f)
}

// read parses one word per line. Trailing whitespace is trimmed (like Python's
// rstrip) and blank lines are skipped (the Python version would keep an empty
// string, which can never match a word).
func read(r io.Reader) (Set, error) {
	set := Set{}
	sc := bufio.NewScanner(r)
	first := true
	for sc.Scan() {
		line := sc.Text()
		if first {
			line = strings.TrimPrefix(line, "\xef\xbb\xbf")
			first = false
		}
		line = strings.TrimRight(line, " \t\r\n\v\f")
		if line == "" {
			continue
		}
		set[line] = struct{}{}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return set, nil
}
