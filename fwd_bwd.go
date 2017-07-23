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
	if len(obs) == 0 {
		res := make(chan map[State]float64)
		close(res)
		return res
	}
	return forwardProbs(newHMMCache(h), h, obs)
}

func forwardProbs(c *hmmCache, h *HMM, obs []Obs) <-chan map[State]float64 {
	res := make(chan map[State]float64, 1)
	go func() {
		defer close(res)
		distribution := newFastStateMapFrom(h, h.Init)
		for _, o := range obs {
			// Compute P(X_i | Z_i)
			emitProbs := h.Emitter.LogProbs(o, h.States...)

			// Compute P(X_0:i, Z_i) from P(X_0:i-1, Z_i)
			outJoints := map[State]float64{}
			distribution.Iter(func(state int, prior float64) {
				prob := prior + emitProbs[state]
				if !math.IsInf(prob, -1) {
					outJoints[h.States[state]] = prob
				}
			})
			res <- outJoints

			// Compute P(X_0:i, Z_i+1) for all Z_i+1
			newDist := newFastStateMap(h)
			for _, trans := range c.Transitions {
				prior, hasPrior := distribution.Get(trans.From)
				if !hasPrior {
					continue
				}
				destProb := prior + trans.Prob + emitProbs[trans.From]
				newDist.AddLog(trans.To, destProb)
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
	if len(obs) == 0 {
		res := make(chan map[State]float64)
		close(res)
		return res
	}
	return backwardProbs(newHMMCache(h), h, obs)
}

func backwardProbs(c *hmmCache, h *HMM, obs []Obs) <-chan map[State]float64 {
	res := make(chan map[State]float64, 1)
	go func() {
		defer close(res)
		distribution := initialBackwardDist(c, h)
		for i := len(obs) - 1; i >= 0; i-- {
			// Compute P(X_i | Z_i)
			emitProbs := h.Emitter.LogProbs(obs[i], h.States...)

			// Compute P(X_i:n | Z_i-1) for all Z_i-1.
			newDist := newFastStateMap(h)
			for _, trans := range c.Transitions {
				nextProb, hasNextProb := distribution.Get(trans.To)
				if !hasNextProb {
					continue
				}
				prob := trans.Prob + nextProb + emitProbs[trans.To]
				newDist.AddLog(trans.From, prob)
			}

			res <- distribution.Map()
			distribution = newDist
		}
	}()
	return res
}

func initialBackwardDist(c *hmmCache, h *HMM) *fastStateMap {
	res := newFastStateMap(h)
	if h.TerminalState == nil {
		for i := range h.States {
			res.Set(i, 0)
		}
		return res
	}
	terminalIdx := c.S2I[h.TerminalState]
	for _, trans := range c.Transitions {
		if trans.To == terminalIdx && !math.IsInf(trans.Prob, -1) {
			res.Set(trans.From, trans.Prob)
		}
	}
	return res
}

// ForwardBackward is a result from the forward-backward.
type ForwardBackward struct {
	// Algorithm inputs.
	HMM *HMM
	Obs []Obs

	// Cached values used for inference.
	ForwardOut  []map[State]float64
	BackwardOut []map[State]float64

	cache *hmmCache
}

// NewForwardBackward creates a Smoother that performs hidden
// state inference given the HMM and the observations.
func NewForwardBackward(h *HMM, obs []Obs) *ForwardBackward {
	res := &ForwardBackward{
		HMM:   h,
		Obs:   obs,
		cache: newHMMCache(h),
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		for fwdOut := range forwardProbs(res.cache, h, obs) {
			res.ForwardOut = append(res.ForwardOut, fwdOut)
		}
		wg.Done()
	}()
	go func() {
		for bwdOut := range backwardProbs(res.cache, h, obs) {
			res.BackwardOut = append(res.BackwardOut, bwdOut)
		}
		wg.Done()
	}()
	wg.Wait()
	return res
}

// LogLikelihood returns the log-likelihood of the output
// sequence.
func (f *ForwardBackward) LogLikelihood() float64 {
	if len(f.Obs) == 0 {
		if f.HMM.TerminalState == nil {
			return 0
		} else {
			if prob, ok := f.HMM.Init[f.HMM.TerminalState]; ok {
				return prob
			} else {
				return math.Inf(-1)
			}
		}
	}

	fwdDist := f.ForwardOut[len(f.ForwardOut)-1]
	bwdDist := f.BackwardOut[0]
	probsSum := math.Inf(-1)
	for state, fwdProb := range fwdDist {
		if bwdProb, ok := bwdDist[state]; ok {
			probsSum = addLogs(probsSum, fwdProb+bwdProb)
		}
	}
	return probsSum
}

// Dist returns the distribution of the hidden state at
// time t.
//
// Each state is mapped to its log probability.
// States with 0 probability are omitted.
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

// CondDist returns the conditional distribution of the
// hidden state at time t, given the all possible hidden
// states at time t-1.
// The result maps states at t-1 to distributions over
// states at time t.
//
// If t is equal to the length of the original sequence,
// then the state after the final state is inferred.
// The solution in this case is trivial when there is a
// terminal state.
//
// The behavior is undefined if the given hidden state has
// zero probability.
func (f *ForwardBackward) CondDist(t int) map[State]map[State]float64 {
	if t == 0 || t > len(f.Obs) {
		panic("time out of bounds")
	}

	// Two distributions that we only need if this is not
	// the state after the final state.
	var emissionDist map[State]float64
	var bwdDist map[State]float64
	if t < len(f.Obs) {
		bwdDist = f.BackwardOut[len(f.BackwardOut)-(t+1)]

		var states []State
		for _, s := range f.HMM.States {
			states = append(states, s)
		}
		probs := f.HMM.Emitter.LogProbs(f.Obs[t], states...)
		emissionDist = map[State]float64{}
		for i, state := range states {
			emissionDist[state] = probs[i]
		}
	}

	prevDist := f.Dist(t - 1)

	res := map[State]map[State]float64{}
	totals := map[State]float64{}
	for state := range prevDist {
		res[state] = map[State]float64{}
		totals[state] = math.Inf(-1)
	}

	// Compute, for each possible Z pair, P(Z, Z_t-1, X).
	// We can compute this using the chain rule as:
	//
	//     P(Z_t-1 | X_0:t-1) P(Z_t | Z_t-1) P(X_t | Z_t) P(X_t+1:n | Z_t)
	//
	// Note that the last two terms do not apply if we are
	// interested in the state after the sequence.
	for trans, transProb := range f.HMM.Transitions {
		fromDist, ok := res[trans.From]
		if !ok {
			continue
		}
		endProb := prevDist[trans.From] + transProb
		if t == len(f.Obs) {
			if f.HMM.TerminalState != nil && trans.To != f.HMM.TerminalState {
				continue
			}
		} else {
			endProb += bwdDist[trans.To] + emissionDist[trans.To]
		}
		totals[trans.From] = addLogs(totals[trans.From], endProb)
		addToState(fromDist, trans.To, endProb)
	}

	// Turn the joints into conditionals.
	for from, total := range totals {
		for to := range res[from] {
			res[from][to] -= total
		}
	}

	return res
}
