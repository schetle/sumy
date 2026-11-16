// Command sumy is a Go port of a slice of the sumy text summarizer. Only the
// Luhn method, plain text input (file or stdin) and the sentence/word
// tokenizers are supported; there is no URL or HTML input.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/miso-belica/sumy/go/internal/stopwords"
	"github.com/miso-belica/sumy/go/internal/summarizer"
	"github.com/miso-belica/sumy/go/internal/tokenizer"
)

// Version is the application version.
const Version = "0.3.0"

const usage = `Sumy - automatic text summarizer.

Usage:
    sumy [luhn] [--length=<length>] [--language=<lang>] [--stopwords=<file_path>] [--file=<file_path>]
    sumy --version
    sumy --help

Options:
    --length=<length>        Length of summarized text. It may be count of sentences
                             or percentage of input text. [default: 20%]
    --language=<lang>        Natural language of summarized text. [default: english]
    --stopwords=<file_path>  Path to a file containing a list of stopwords. One word per line in UTF-8 encoding.
                             If it's not provided default list of stop-words is used according to chosen language.
    --file=<file_path>       Path to the text file to summarize. Standard input is read when absent.
    --version                Displays current application version.
    --help                   Displays this text.
`

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

// run executes the CLI and returns the process exit code: 0 on success, 1 on
// any error (reported on stderr).
func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("sumy", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {}
	version := fs.Bool("version", false, "Displays current application version.")
	length := fs.String("length", "20%", "Length of summarized text (count or percentage).")
	language := fs.String("language", "english", "Natural language of summarized text.")
	stopFile := fs.String("stopwords", "", "Path to a file with stop words.")
	file := fs.String("file", "", "Path to the text file to summarize.")

	// Flags may appear before and after the optional "luhn" positional
	// (flag.Parse stops at the first non-flag argument).
	method := ""
	rest := args
	for {
		if err := fs.Parse(rest); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				fmt.Fprint(stdout, usage)
				return 0
			}
			fmt.Fprint(stderr, usage)
			return 1
		}
		if fs.NArg() == 0 {
			break
		}
		if method != "" || fs.Arg(0) != "luhn" {
			fmt.Fprintf(stderr, "sumy: unsupported argument %q (only the \"luhn\" method is available)\n", fs.Arg(0))
			return 1
		}
		method = fs.Arg(0)
		rest = fs.Args()[1:]
	}

	if *version {
		fmt.Fprintln(stdout, Version)
		return 0
	}
	if err := summarize(*length, *language, *stopFile, *file, stdin, stdout); err != nil {
		fmt.Fprintf(stderr, "sumy: %v\n", err)
		return 1
	}
	return 0
}

func summarize(lengthArg, language, stopFile, file string, stdin io.Reader, stdout io.Writer) error {
	length, err := summarizer.ParseLength(lengthArg)
	if err != nil {
		return err
	}

	var stop stopwords.Set
	if stopFile != "" {
		stop, err = stopwords.ReadFile(stopFile)
	} else {
		stop, err = stopwords.Get(language)
	}
	if err != nil {
		return err
	}

	tk, err := tokenizer.New(language)
	if err != nil {
		return err
	}

	var input []byte
	if file != "" {
		input, err = os.ReadFile(file)
	} else {
		input, err = io.ReadAll(stdin)
	}
	if err != nil {
		return err
	}

	l := summarizer.NewLuhn()
	l.SetStopWords(stop)
	doc := summarizer.ParsePlaintext(string(input), tk)
	for _, s := range l.Summarize(doc, length) {
		// One sentence per line: collapse any inner line breaks.
		fmt.Fprintln(stdout, strings.Join(strings.Fields(s.Text), " "))
	}
	return nil
}
