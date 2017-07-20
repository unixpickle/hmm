package hmm

import "math"

// ForwardProbs computes, for each timestep i, for each
// state y, the joint probability P(x,y) where x is the
// first (i+1) observations.
//
// Probabilities are experssed in the log domain.
// States with 0 probability are omitted.
//
// The caller should read through the entire channel,
// which is fed len(obs) items.
// The caller may modify the returned maps.
func ForwardProbs(h *HMM, obs []Obs) <-chan map[State]float64 {
	res := make(chan map[State]float64, 1)
	distribution := h.Init
	go func() {
		defer close(res)
		for _, o := range obs {
			// Compute P(X_i | Z_i)
			emitProbs := h.Emitter.LogProbs(o, h.States...)
			emitProbsMap := map[State]float64{}
			for i, state := range h.States {
				emitProbsMap[state] = emitProbs[i]
			}

			// Compute P(X_0:i, Z_i) from P(X_0:i-1, Z_i)
			outJoints := map[State]float64{}
			for state, prior := range distribution {
				prob := prior + emitProbsMap[state]
				if !math.IsInf(prob, -1) {
					outJoints[state] = prob
				}
			}
			res <- outJoints

			// Compute P(X_0:i, Z_i+1)
			newDist := map[State]float64{}
			for trans, transProb := range h.Transitions {
				prior, hasPrior := distribution[trans.From]
				if !hasPrior {
					continue
				}
				destProb := prior + transProb + emitProbsMap[trans.From]
				if !math.IsInf(destProb, -1) {
					if oldProb, ok := newDist[trans.To]; ok {
						newDist[trans.To] = addLogs(oldProb, destProb)
					} else {
						newDist[trans.To] = destProb
					}
				}
			}
			distribution = newDist
		}
	}()
	return res
}
