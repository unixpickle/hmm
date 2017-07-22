package hmm

import "math"

// BaumWelch applies a step of the BaumWelch algorithm to
// the discrete-observation HMM.
// It returns a new *HMM with some modified fields which
// reflect the updated parameters.
//
// The HMM must use a TabularEmitter.
func BaumWelch(h *HMM, data <-chan []Obs) *HMM {
	initTally := map[State]float64{}
	initTotal := math.Inf(-1)

	transTally := map[Transition]float64{}
	fromStateTotals := map[State]float64{}

	emitTally := TabularEmitter{}
	emitTotals := map[State]float64{}

	for sample := range data {
		if len(sample) == 0 {
			if h.TerminalState != nil {
				initTotal = addLogs(initTotal, 0)
				addToState(initTally, h.TerminalState, 0)
			}
			continue
		}

		fb := NewForwardBackward(h, sample)

		initTotal = addLogs(initTotal, 0)
		for state, prob := range fb.Dist(0) {
			addToState(initTally, state, prob)
		}

		for t := 1; t <= len(sample); t++ {
			if t == len(sample) && h.TerminalState == nil {
				// Looking at the final state transitions don't make
				// much sense, since we really have no idea what the
				// next state should be.
				break
			}

			prevDist := fb.Dist(t - 1)
			for state, prob := range prevDist {
				addToState(fromStateTotals, state, prob)
			}
			condDist := fb.CondDist(t)
			for from, tos := range condDist {
				for to, condProb := range tos {
					joint := condProb + prevDist[from]
					trans := Transition{From: from, To: to}
					if oldProb, ok := transTally[trans]; ok {
						transTally[trans] = addLogs(oldProb, joint)
					} else {
						transTally[trans] = joint
					}
				}
			}
		}

		for t, obs := range sample {
			for state, prob := range fb.Dist(t) {
				if _, ok := emitTally[state]; !ok {
					emitTally[state] = map[Obs]float64{}
				}
				addToState(emitTotals, state, prob)
				emissions := emitTally[state]
				if oldProb, ok := emissions[obs]; ok {
					emissions[obs] = addLogs(oldProb, prob)
				} else {
					emissions[obs] = prob
				}
			}
		}
	}

	// Normalize probabilities.
	for trans := range transTally {
		transTally[trans] -= fromStateTotals[trans.From]
	}
	for state := range initTally {
		initTally[state] -= initTotal
	}
	for state, emissions := range emitTally {
		total := emitTotals[state]
		for obs := range emissions {
			emissions[obs] -= total
		}
	}

	res := *h
	res.Emitter = emitTally
	res.Transitions = transTally
	res.Init = initTally
	return &res
}
