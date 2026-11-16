// Command sumy is a Go port of a slice of the sumy text summarizer.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
)

// Version is the application version.
const Version = "0.1.0"

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("sumy", flag.ContinueOnError)
	fs.SetOutput(stderr)
	version := fs.Bool("version", false, "Displays current application version.")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if *version {
		fmt.Fprintln(stdout, Version)
		return 0
	}
	return 0
}
