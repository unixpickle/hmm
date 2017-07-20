package anyhmm

import "math/rand"

func sampleIndex(gen *rand.Rand, probs []float64) int {
	if len(probs) == 0 {
		panic("cannot sample from empty lits")
	}
	var offset float64
	if gen == nil {
		offset = rand.Float64()
	} else {
		offset = gen.Float64()
	}
	for i, p := range probs {
		offset -= p
		if offset < 0 {
			return i
		}
	}
	return len(probs) - 1
}
