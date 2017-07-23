package hmm

import "math"

// hmmCache caches a fast transition matrix and a state to
// index mapping.
type hmmCache struct {
	S2I         map[State]int
	Transitions []fastTransition
}

func newHMMCache(h *HMM) *hmmCache {
	s2i := statesToIndices(h)
	return &hmmCache{
		S2I:         s2i,
		Transitions: fastTransitions(h, s2i),
	}
}

// statesToIndices creates a mapping from states to their
// indices in an HMM.
func statesToIndices(h *HMM) map[State]int {
	res := map[State]int{}
	for i, state := range h.States {
		res[state] = i
	}
	return res
}

// fastTransition is used to cache information about a
// Transition.
// It uses state indices rather than actual states.
type fastTransition struct {
	From int
	To   int
	Prob float64
}

// fastTransitions converts the model's transition matrix
// to a more efficient-to-use format.
func fastTransitions(h *HMM, s2i map[State]int) []fastTransition {
	res := make([]fastTransition, 0, len(h.Transitions))
	for trans, prob := range h.Transitions {
		res = append(res, fastTransition{
			From: s2i[trans.From],
			To:   s2i[trans.To],
			Prob: prob,
		})
	}
	return res
}

// fastStateMap is an alternative to map[State]float64
// that uses slices rather than maps.
type fastStateMap struct {
	h       *HMM
	values  []float64
	present []bool
}

// newFastStateMap creates an empty fastStateMap.
func newFastStateMap(h *HMM) *fastStateMap {
	return &fastStateMap{
		h:       h,
		values:  make([]float64, len(h.States)),
		present: make([]bool, len(h.States)),
	}
}

// newFastStateMapFrom creates a fastStateMap from a
// native state map.
func newFastStateMapFrom(h *HMM, m map[State]float64) *fastStateMap {
	res := newFastStateMap(h)
	for i, state := range h.States {
		res.values[i], res.present[i] = m[state]
	}
	return res
}

// Get gets the value for the given state.
func (f *fastStateMap) Get(state int) (val float64, hasVal bool) {
	return f.values[state], f.present[state]
}

// Set sets the value for the given state.
func (f *fastStateMap) Set(state int, val float64) {
	f.present[state] = true
	f.values[state] = val
}

// AddLog adds the logarithm to the entry for the state.
// No entries are added for -infinity values.
func (f *fastStateMap) AddLog(state int, x float64) {
	if !math.IsInf(x, -1) {
		if f.present[state] {
			f.values[state] = addLogs(f.values[state], x)
		} else {
			f.values[state] = x
			f.present[state] = true
		}
	}
}

// AddAll adds the value to every entry.
func (f *fastStateMap) AddAll(x float64) {
	for i, present := range f.present {
		if present {
			f.values[i] += x
		}
	}
}

// Iter iterates over the entries of f.
func (f *fastStateMap) Iter(handler func(state int, val float64)) {
	for i, present := range f.present {
		if present {
			handler(i, f.values[i])
		}
	}
}

// Map converts f into a native map.
func (f *fastStateMap) Map() map[State]float64 {
	res := map[State]float64{}
	for i, present := range f.present {
		if present {
			res[f.h.States[i]] = f.values[i]
		}
	}
	return res
}
