package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

const (
	confDirectory string = "data"
	confExtension string = ".txt"
	confOutput    string = "output.txt"
)

func main() {
	// initialization
	filesList, err := filesList(confDirectory, confExtension)
	if err != nil {
		fmt.Println("@ Error reading directory: %v", err)
		return
	}

	clustering := NewClustering()
    for _, file := range filesList {
		// define clusters
		name := file[len(confDirectory) + 1:]
		index := strings.Index(name, "/")
		if index == -1 {
			fmt.Println("@ Error during clustering subdirectories")
			return
		}
		cluster := name[:index]
		clustering.appendCluster(cluster)

		// read all words from file
		words, err := readWordsFromFile(file)
		if err != nil {
			fmt.Println("@ Error reading file %s: %v", file, err)
			continue
		}

		// add words to bag of words
		clustering.appendWords(words)

		// assign read words to documents
		document := NewDocument()
		document.set(name, words)
		clustering.appendDocument(name, document)

		// assign one document per cluster
		clustering.appendGroup(cluster, document)
	}

	timeStart := time.Now()
	// expectation-maximization algorithm
	probDocInCluster, iterations := clustering.em()
	timeEnd := time.Now()	

	// calculate & print statistics
	fmt.Println("# Start time:", timeStart.Format("2006-01-02 15:04:05"))
	fmt.Println("# End time:", timeEnd.Format("2006-01-02 15:04:05"))
	fmt.Println("# Initial parametrs:")
	for group := range(clustering.Groups) {
		fmt.Println("  >", group, ":", clustering.Groups[group].Name)
	}

	fmt.Println("# Iterations:", iterations)
	clustersDistribution := make(map[string]int)
	for _, cluster := range(clustering.Clusters) {
		clustersDistribution[cluster] = 0
	}

	notClusterized := 0
	for document := range(probDocInCluster) {
		maxValue := 0.0
		for _, cluster := range(clustering.Clusters) {
			number, _ := probDocInCluster[document][cluster].Float64()
			if number > maxValue {
				maxValue = number
			}
		}
		for _, cluster := range(clustering.Clusters) {
			if maxValue == 0.0 {
				notClusterized += 1
			}
			number, _ := probDocInCluster[document][cluster].Float64()
			if number == maxValue {
				clustersDistribution[cluster] += 1
			}
		}
	}

	fmt.Println("# Clusters distribution:")
	clusterized := 0
	for _, cluster := range(clustering.Clusters) {
		clusterized += clustersDistribution[cluster]
		fmt.Println("  >", cluster, ":", clustersDistribution[cluster])
	}
	fmt.Println("# Documents not clusterized:", notClusterized)
	fmt.Println("# Documents clusterized:", clusterized)

	// write to file
	fileOutput, err := os.Create(confOutput)
	if err != nil {
		fmt.Println("@ Error during creating output file")
	}

	writerOutput := bufio.NewWriter(fileOutput)
	for document := range(probDocInCluster) {
		for _, cluster := range(clustering.Clusters) {
			number, _ := probDocInCluster[document][cluster].Float64()
			if number > 0.1 {
				line := fmt.Sprintf("%s -> %s : %f", document, cluster, number)
				fmt.Fprintln(writerOutput, line)
			}
		}
	}
	writerOutput.Flush()
	fileOutput.Close()
	fmt.Println("# Data saved to files!")
}

func readWordsFromFile(path string) ([]string, error) {
	b, err := ioutil.ReadFile(path)
    if err != nil {
        return nil, err
	}

	content := string(b)
	re := regexp.MustCompile("[a-zA-Z]+")
	return re.FindAllString(content, -1), nil
}

func filesList(directory, pattern string) ([]string, error) {
	filesList := []string{}
	err := filepath.Walk(directory, func(path string, info os.FileInfo, err error) error {
		if info.IsDir() {
			return nil
		}

		if strings.HasSuffix(path, pattern) {
			filesList = append(filesList, path)
		}
		return nil
	})
	return filesList, err
}