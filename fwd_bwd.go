package hmm

import "math"

// ForwardProbs computes, for each timestep i, for each
// state y, the joint probability P(x,y) where x is the
// first (i+1) observations.
//
// Probabilities are experssed in the log domain.
// States with 0 probability are omitted.
//
// The caller should not modify the returned maps.
//
// The caller should read through the entire channel,
// which is fed len(obs) items.
func ForwardProbs(h *HMM, obs []Obs) <-chan map[State]float64 {
	res := make(chan map[State]float64, 1)
	distribution := h.Init
	go func() {
		defer close(res)
		for _, o := range obs {
			newDist := map[State]float64{}
			emitProbs := h.Emitter.LogProbs(o, h.States)
			emitProbsMap := map[State]float64{}
			for i, state := range h.States {
				emitProbsMap[state] = emitProbs[i]
			}
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
			res <- newDist
			distribution = newDist
		}
	}()
	return res
}

func addLogs(x1, x2 float64) float64 {
	max := math.Max(x1, x2)
	if math.IsInf(max, -1) {
		return max
	}
	return math.Log(math.Exp(x1-max)+math.Exp(x2-max)) + max
}
