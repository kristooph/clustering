package main

import (
	"math"
	"math/big"
)

type Clustering struct {
	BagOfWords map[string]struct{}  // bag of words
	Clusters   []string             // clusters
	Documents  map[string]*Document // documents
	Groups     map[string]*Document // initial assign document per cluster
}

func NewClustering() *Clustering {
	clustering := &Clustering{}
	clustering.BagOfWords = make(map[string]struct{})
	clustering.Documents = make(map[string]*Document)
	clustering.Groups = make(map[string]*Document)
	return clustering
}

// append cluster to clusters if not exists
func (clustering *Clustering) appendCluster(cluster string) {
	for _, element := range clustering.Clusters {
        if element == cluster {
            return
        }
	}
	clustering.Clusters = append(clustering.Clusters, cluster)
}

func (clustering *Clustering) appendDocument(name string, document *Document) {
	clustering.Documents[name] = document
}

func (clustering *Clustering) appendGroup(group string, document *Document) {
	if clustering.groupExists(group) {
		return
	}
	clustering.Groups[group] = document
}

func (clustering *Clustering) appendWords(words []string) {
	for _, word := range(words) {
		clustering.BagOfWords[word] = struct{}{}
	}
}

func (clustering *Clustering) groupExists(group string) bool {
	_, exists := clustering.Groups[group]
	return exists
}

func (clustering *Clustering) em() (map[string]map[string]*big.Float, int) {
	const (
		convergence = 1.0e-4 // convergence value
		epsilon     = 1.0e-4 // epsilon
		prec        = 400    // big.Float precision
	)

	// initialize clusters probability
	probCluster := make(map[string]*big.Float)
	for _, cluster := range(clustering.Clusters) {
		probCluster[cluster] = new(big.Float).SetPrec(prec).SetFloat64(1 / float64(len(clustering.Clusters)))
	}

	// initialize documents in clusters probability
	probDocInCluster := make(map[string]map[string]*big.Float)
	for _, document := range(clustering.Documents) {
		if _, exists := probDocInCluster[document.Name]; !exists {
			probDocInCluster[document.Name] = make(map[string]*big.Float)
		}
		for _, cluster := range(clustering.Clusters) {
			if document.Name == clustering.Groups[cluster].Name {
				probDocInCluster[document.Name][cluster] = new(big.Float).SetPrec(prec).SetFloat64(1.0)
				continue
			}
			probDocInCluster[document.Name][cluster] = new(big.Float).SetPrec(prec).SetFloat64(0.0)
		}
	}

	// initialize words in clusters probability
	probWordInCluster := make(map[string]map[string]*big.Float)
	for word := range(clustering.BagOfWords) {
		if _, exists := probWordInCluster[word]; !exists {
			probWordInCluster[word] = make(map[string]*big.Float)
		}
		for _, cluster := range(clustering.Clusters) {
			probWordInCluster[word][cluster] = new(big.Float).SetPrec(prec).SetFloat64(0.0)
		}
	}

	// variables initialization
	var (
		iteration        int        = 1   // current iteration
		iterations       int        = 100 // number of max iterations
		likeHoodCurrent  float64    = 0.0 // (t)-iteration
		likeHoodPrevious float64    = 0.0 // (t-1)-iteration
	)

	var (
		denominator  *big.Float = new(big.Float).SetPrec(prec)                 // denominator used in E, M steps
		nominator	 *big.Float = new(big.Float).SetPrec(prec)                 // nominator used in E, M step
		normalizator *big.Float = new(big.Float).SetPrec(prec)                 // normalizator used after M step
		zero         *big.Float = new(big.Float).SetPrec(prec).SetFloat64(0.0) // zero represented by big.Float
	)

	denominators := make(map[string]*big.Float) // denominators used i E step
	for _, document := range(clustering.Documents) {
		denominators[document.Name] = new(big.Float).SetPrec(prec)
		for _, cluster := range(clustering.Clusters) {
			number, _ := probDocInCluster[document.Name][cluster].Float64()
			if number > 0 {
				likeHoodCurrent += math.Log(number)
			}
		}
	}

	// EM algorithm
	for ; iteration < iterations; iteration += 1 {
		likeHoodPrevious = likeHoodCurrent

		// maximization step
		for word := range(clustering.BagOfWords) {
			for _, cluster := range(clustering.Clusters) {
				nominator.SetFloat64(epsilon)
				denominator.SetFloat64(0.0)
				for _, document := range(clustering.Documents) {
					if document.contains(word) {
						nominator.Add(nominator, probDocInCluster[document.Name][cluster])
					}
					denominator.Add(denominator, probDocInCluster[document.Name][cluster])
				}
				probWordInCluster[word][cluster].Quo(nominator, denominator)
			}
		}

		for _, cluster := range(clustering.Clusters) {
			probCluster[cluster].SetFloat64(0.0)
			for _, document := range(clustering.Documents) {
				probCluster[cluster].Add(probCluster[cluster], probDocInCluster[document.Name][cluster])
			}
		}

		// normalization
		normalizator.SetFloat64(0.0)
		for _, cluster := range(clustering.Clusters) {
			normalizator.Add(normalizator, probCluster[cluster])
		}
		for _, cluster := range(clustering.Clusters) {
			probCluster[cluster].Quo(probCluster[cluster], normalizator) // quo is divide method for floats
		}
		
		// expectation step
		for _, document := range(clustering.Documents) {
			denominators[document.Name].SetFloat64(0.0)
			for _, cluster := range(clustering.Clusters) {
				nominatorE := new(big.Float).SetPrec(prec)
				nominatorE.Set(probCluster[cluster])
				for word := range(document.Words) {
					if probWordInCluster[word][cluster].Cmp(zero) == 1 { // cmp method returns zero if x > y
						nominatorE.Mul(nominatorE, probWordInCluster[word][cluster])
					}
				}
				probDocInCluster[document.Name][cluster] = nominatorE
				denominators[document.Name].Add(denominators[document.Name], nominatorE)
			}
		}

		for _, document := range(clustering.Documents) {
			for _, cluster := range(clustering.Clusters) {
				probDocInCluster[document.Name][cluster].Quo(probDocInCluster[document.Name][cluster], denominators[document.Name])
			}
		}

		// convergence check
		likeHoodCurrent = 0.0
		for _, document := range(clustering.Documents) {
			for _, cluster := range(clustering.Clusters) {
				number, _ := probDocInCluster[document.Name][cluster].Float64()
				if number > 0 {
					likeHoodCurrent += math.Log(number)
				}
			}
		}

		if math.Abs(likeHoodCurrent - likeHoodPrevious) < convergence {
			break
		}
	}

	return probDocInCluster, iteration
}
