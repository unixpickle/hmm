package hmm

import "math"

// MostLikely returns the most probable sequence of states
// given the observation sequence.
//
// If no hidden sequence can explain the observations, nil
// is returned.
func MostLikely(h *HMM, obs []Obs) []State {
	// Maps the final state of a path to that path.
	paths := map[State]*viterbiPath{}

	for state, logProb := range h.Init {
		paths[state] = &viterbiPath{
			Seq:     []State{state},
			LogProb: logProb,
		}
	}

	for i, o := range obs {
		viterbiObservation(h, o, paths)
		if h.TerminalState != nil || i+1 < len(obs) {
			paths = viterbiTransition(h, paths)
		}
	}

	if h.TerminalState != nil {
		if path, ok := paths[h.TerminalState]; ok {
			return path.Seq[:len(path.Seq)-1]
		}
		return nil
	}

	var mostLikely *viterbiPath
	bestProb := math.Inf(-1)
	for _, path := range paths {
		if path.LogProb >= bestProb {
			mostLikely = path
			bestProb = path.LogProb
		}
	}

	if mostLikely == nil {
		return nil
	}
	return mostLikely.Seq
}

func viterbiObservation(h *HMM, obs Obs, paths map[State]*viterbiPath) {
	var states []State
	for state := range paths {
		states = append(states, state)
	}
	emitProbs := h.Emitter.LogProbs(obs, states...)
	for i, state := range states {
		path := paths[state]
		path.LogProb += emitProbs[i]
		if math.IsInf(path.LogProb, -1) {

		}
	}
}

func viterbiTransition(h *HMM, paths map[State]*viterbiPath) map[State]*viterbiPath {
	res := map[State]*viterbiPath{}
	for trans, transProb := range h.Transitions {
		oldPath, ok := paths[trans.From]
		if !ok {
			continue
		}
		newProb := transProb + oldPath.LogProb
		if math.IsInf(newProb, -1) {
			continue
		}
		if existing, ok := res[trans.To]; !ok || existing.LogProb < newProb {
			res[trans.To] = &viterbiPath{
				Seq:     append(append([]State{}, oldPath.Seq...), trans.To),
				LogProb: newProb,
			}
		}
	}
	return res
}

type viterbiPath struct {
	Seq     []State
	LogProb float64
}
