package hmm

import (
	"math"
	"math/rand"
	"reflect"

	"github.com/unixpickle/serializer"
)

// sampleIndex samples an index from the list, given the
// probability of each index.
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

// addLogs adds two numbers in the log domain.
func addLogs(x1, x2 float64) float64 {
	max := math.Max(x1, x2)
	if math.IsInf(max, -1) {
		return max
	}
	return math.Log(math.Exp(x1-max)+math.Exp(x2-max)) + max
}

// addToState adds the given probability to the entry for
// the given state in the map.
// Entries are added to the map as needed.
//
// If prob is -infinity, this is a no-op.
func addToState(m map[State]float64, s State, prob float64) {
	if math.IsInf(prob, -1) {
		return
	}
	if lastProb, ok := m[s]; ok {
		m[s] = addLogs(lastProb, prob)
	} else {
		m[s] = prob
	}
}

// randomDist generates a random probability distribution.
//
// The probabilities are expressed in the log domain.
func randomDist(gen *rand.Rand, n int) []float64 {
	var res []float64
	sum := math.Inf(-1)
	for i := 0; i < n; i++ {
		var val float64
		if gen != nil {
			val = gen.NormFloat64()
		} else {
			val = rand.NormFloat64()
		}
		sum = addLogs(sum, val)
		res = append(res, val)
	}
	for i := range res {
		res[i] -= sum
	}
	return res
}

func serializersComparable(slices ...[]serializer.Serializer) bool {
	for _, slice := range slices {
		for _, item := range slice {
			if !reflect.TypeOf(item).Comparable() {
				return false
			}
		}
	}
	return true
}
