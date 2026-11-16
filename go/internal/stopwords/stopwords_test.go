package stopwords

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGetAllLanguages(t *testing.T) {
	for _, lang := range []string{"czech", "english", "french", "german", "slovak"} {
		s, err := Get(lang)
		if err != nil {
			t.Fatalf("Get(%q): %v", lang, err)
		}
		if len(s) < 10 {
			t.Errorf("Get(%q): only %d words", lang, len(s))
		}
	}
}

func TestGetEnglishContents(t *testing.T) {
	s, err := Get("english")
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []string{"the", "a", "and"} {
		if !s.Contains(w) {
			t.Errorf("english stop words missing %q", w)
		}
	}
	if s.Contains("") || s.Contains("summarizer") {
		t.Error("unexpected word in english stop words")
	}
}

func TestGetUnknownLanguage(t *testing.T) {
	for _, lang := range []string{"klingon", "", "../english", "english/x"} {
		if _, err := Get(lang); err == nil {
			t.Errorf("Get(%q): expected error", lang)
		}
	}
}

func TestReadFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "sw.txt")
	content := "\xef\xbb\xbfjá\r\njsem  \n\nčeština\n"
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(s) != 3 || !s.Contains("já") || !s.Contains("jsem") || !s.Contains("čeština") {
		t.Errorf("unexpected set: %v", s)
	}
}

func TestReadFileMissing(t *testing.T) {
	if _, err := ReadFile(filepath.Join(t.TempDir(), "nope.txt")); err == nil {
		t.Error("expected error")
	}
}
