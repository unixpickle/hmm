package hmm

import (
	"math"
	"math/rand"
)

func sampleIndex(gen *rand.Rand, probs []float64) int {
	if len(probs) == 0 {
		panic("cannot sample from empty list")
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

func addLogs(x1, x2 float64) float64 {
	max := math.Max(x1, x2)
	if math.IsInf(max, -1) {
		return max
	}
	return math.Log(math.Exp(x1-max)+math.Exp(x2-max)) + max
}
