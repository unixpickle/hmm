package hmm

import (
	"math"
	"sync"
)

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
	go func() {
		defer close(res)
		distribution := h.Init
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

			// Compute P(X_0:i, Z_i+1) for all Z_i+1
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

// BackwardProbs computes, for each timestep i, the
// probability of the observations after timestep i given
// each hidden state at timestep i.
//
// The timesteps in the channel are in reverse order,
// starting at the last timestep going backward.
//
// See ForwardProbs for more on how to use the resulting
// channel.
func BackwardProbs(h *HMM, obs []Obs) <-chan map[State]float64 {
	res := make(chan map[State]float64, 1)
	go func() {
		defer close(res)
		distribution := initialBackwardDist(h)
		for i := len(obs) - 1; i >= 0; i-- {
			// Compute P(X_i | Z_i)
			emitProbs := h.Emitter.LogProbs(obs[i], h.States...)
			emitProbsMap := map[State]float64{}
			for i, state := range h.States {
				emitProbsMap[state] = emitProbs[i]
			}

			// Compute P(X_i:n | Z_i-1) for all Z_i-1.
			newDist := map[State]float64{}
			for trans, transProb := range h.Transitions {
				nextProb, hasNextProb := distribution[trans.To]
				if !hasNextProb {
					continue
				}
				prob := transProb + nextProb + emitProbsMap[trans.To]
				if !math.IsInf(prob, -1) {
					if lastProb, ok := newDist[trans.From]; ok {
						newDist[trans.From] = addLogs(lastProb, prob)
					} else {
						newDist[trans.From] = prob
					}
				}
			}

			res <- distribution
			distribution = newDist
		}
	}()
	return res
}

func initialBackwardDist(h *HMM) map[State]float64 {
	res := map[State]float64{}
	if h.TerminalState == nil {
		for _, state := range h.States {
			res[state] = 0
		}
		return res
	}
	for trans, prob := range h.Transitions {
		if trans.To == h.TerminalState {
			res[trans.From] = prob
		}
	}
	return res
}

// ForwardBackward is a result from the forward-backward.
type ForwardBackward struct {
	// Algorithm inputs.
	HMM *HMM
	Obs []Obs

	// Algorithm byproducts (cached values used for
	// inference).
	ForwardOut  []map[State]float64
	BackwardOut []map[State]float64
}

// NewForwardBackward creates a Smoother that performs hidden
// state inference given the HMM and the observations.
func NewForwardBackward(h *HMM, obs []Obs) *ForwardBackward {
	res := &ForwardBackward{}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		for fwdOut := range ForwardProbs(h, obs) {
			res.ForwardOut = append(res.ForwardOut, fwdOut)
		}
		wg.Done()
	}()
	go func() {
		for bwdOut := range BackwardProbs(h, obs) {
			res.BackwardOut = append(res.BackwardOut, bwdOut)
		}
		wg.Done()
	}()
	wg.Wait()
	return res
}

// Dist returns the distribution of the hidden state at
// time t.
// Each state is mapped to its log probability.
// States with zero probability may be absent.
func (f *ForwardBackward) Dist(t int) map[State]float64 {
	res := map[State]float64{}
	bwdDist := f.BackwardOut[len(f.BackwardOut)-(t+1)]
	fwdDist := f.ForwardOut[t]
	probsSum := math.Inf(-1)
	for state, fwdProb := range fwdDist {
		if bwdProb, ok := bwdDist[state]; ok {
			prob := fwdProb + bwdProb
			res[state] = prob
			probsSum = addLogs(probsSum, prob)
		}
	}

	// Divide by P(X1,...,Xn).
	for state := range res {
		res[state] -= probsSum
	}

	return res
}
