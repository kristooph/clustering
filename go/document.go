package main

type Document struct {
	Name  string              // document name
	Words map[string]struct{} // words in document
}

func NewDocument() *Document {
	document := &Document{}
	document.Words = make(map[string]struct{})
	return document
}

func (document *Document) set(name string, words []string) {
	document.Name = name
	for _, word := range(words) {
		document.Words[word] = struct{}{}
	}
}

func (document *Document) contains(word string) bool {
	_, exists := document.Words[word]
	return exists
}