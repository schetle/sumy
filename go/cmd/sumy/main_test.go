package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func readTestdata(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// Golden tests: each case runs the CLI on testdata input and compares stdout
// with the expected file. When stdin is true the input is piped instead of
// being passed with --file.
func TestGolden(t *testing.T) {
	tests := []struct {
		name     string
		args     []string
		input    string // testdata file
		stdin    bool
		expected string // testdata file
	}{
		{"english default length (20%)", nil, "english_input.txt", true, "english_default.expected"},
		{"english luhn positional with count", []string{"luhn", "--length", "2"}, "english_input.txt", false, "english_2.expected"},
		{"english flags after positional, percent", []string{"luhn", "--length=50%"}, "english_input.txt", true, "english_50pct.expected"},
		{"czech language", []string{"--language", "czech", "--length", "2"}, "czech_input.txt", false, "czech_2.expected"},
		{"slovak alias on czech text", []string{"--language=slovak", "--length=1"}, "czech_input.txt", true, "slovak_1.expected"},
		{"custom stopwords", []string{"--language", "czech", "--length", "2", "--stopwords", "testdata/custom_stopwords.txt"}, "czech_input.txt", false, "czech_custom_stopwords.expected"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			args := append([]string(nil), tc.args...)
			var stdin *strings.Reader = strings.NewReader("")
			if tc.stdin {
				stdin = strings.NewReader(readTestdata(t, tc.input))
			} else {
				args = append(args, "--file", filepath.Join("testdata", tc.input))
			}
			var out, errb bytes.Buffer
			if code := run(args, stdin, &out, &errb); code != 0 {
				t.Fatalf("exit %d, stderr: %s", code, errb.String())
			}
			if errb.Len() != 0 {
				t.Errorf("unexpected stderr: %s", errb.String())
			}
			if want := readTestdata(t, tc.expected); out.String() != want {
				t.Errorf("output mismatch\ngot:\n%s\nwant:\n%s", out.String(), want)
			}
		})
	}
}

func TestVersionAndHelp(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run([]string{"--version"}, strings.NewReader(""), &out, &errb); code != 0 || out.String() != Version+"\n" {
		t.Errorf("version: code %d out %q", code, out.String())
	}
	out.Reset()
	if code := run([]string{"--help"}, strings.NewReader(""), &out, &errb); code != 0 || !strings.Contains(out.String(), "Usage:") {
		t.Errorf("help: code %d out %q", code, out.String())
	}
}

func TestEmptyInput(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run(nil, strings.NewReader("  \n"), &out, &errb); code != 0 || out.Len() != 0 {
		t.Errorf("code %d out %q err %q", code, out.String(), errb.String())
	}
}

func TestErrors(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"unknown language", []string{"--language", "klingon"}, "klingon"},
		{"bad length", []string{"--length", "abc"}, "invalid length"},
		{"missing file", []string{"--file", "testdata/does-not-exist.txt"}, "does-not-exist"},
		{"missing stopwords file", []string{"--stopwords", "testdata/nope.txt"}, "nope.txt"},
		{"unknown method", []string{"lsa"}, "unsupported"},
		{"unknown flag", []string{"--bogus"}, "bogus"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out, errb bytes.Buffer
			code := run(tc.args, strings.NewReader("Some text. More text."), &out, &errb)
			if code != 1 {
				t.Errorf("exit code %d, want 1", code)
			}
			if out.Len() != 0 {
				t.Errorf("unexpected stdout %q", out.String())
			}
			if !strings.Contains(errb.String(), tc.want) {
				t.Errorf("stderr %q does not contain %q", errb.String(), tc.want)
			}
		})
	}
}
