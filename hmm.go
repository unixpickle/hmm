package anyhmm

import (
	"fmt"
	"math"
	"math/rand"
)

// State is a discrete state in an HMM.
// States must be comparable with the == operator.
type State interface{}

// Transition represents the transition between a start
// state and an end state.
type Transition struct {
	From State
	To   State
}

// HMM is a hidden Markov model.
type HMM struct {
	// States contains all allowed states.
	States []State

	// TerminalState is the state that signals the end of
	// an observation chain.
	// Terminal states do not produce an emission, and are
	// implied by the end of the sequence.
	//
	// If TerminalState is nil, then probabilities are
	// computed without accounting for termination.
	TerminalState State

	// Emitter provides emission probabilities.
	Emitter Emitter

	// Init stores the initial state distribution.
	// It maps states to log probabilities.
	// If a state is absent, it has 0 probability.
	Init map[State]float64

	// Transitions stores the log probability for every
	// allowed transition.
	// If a transition is absent, it has 0 probability.
	Transitions map[Transition]float64
}

// Sample samples a sequence of observations and hidden
// states from the model.
//
// Sample requires that h.TerminalState is set.
// Otherwise, the sequence would go on forever and no
// sample would be complete.
//
// If gen is not nil, it may be used instead of the
// global routines in package rand.
// However, even gen is not nil, there is no guarantee
// that it will be used consistently across runs.
func (h *HMM) Sample(gen *rand.Rand) ([]State, []Obs) {
	if h.TerminalState == nil {
		panic("cannot sample without a terminal state")
	}

	var states []State
	var obs []Obs

	state := h.sampleStart(gen)

	var ts *transSampler
	for state != h.TerminalState {
		states = append(states, state)
		obs = append(obs, h.Emitter.Sample(gen, state))
		if ts == nil {
			ts = newTransSampler(h.States, h.Transitions)
		}
		state = ts.Sample(gen, state)
	}

	return states, obs
}

func (h *HMM) sampleStart(gen *rand.Rand) State {
	var states []State
	var probs []float64
	for state, logProb := range h.Init {
		states = append(states, state)
		probs = append(probs, math.Exp(logProb))
	}
	return states[sampleIndex(gen, probs)]
}

type transSampler struct {
	Targets map[State][]State
	Probs   map[State][]float64
}

func newTransSampler(states []State, trans map[Transition]float64) *transSampler {
	res := &transSampler{}
	for tr, logProb := range trans {
		res.Targets[tr.From] = append(res.Targets[tr.From], tr.To)
		res.Probs[tr.From] = append(res.Probs[tr.From], math.Exp(logProb))
	}
	return res
}

func (t *transSampler) Sample(gen *rand.Rand, from State) State {
	if len(t.Targets[from]) == 0 {
		panic(fmt.Sprintf("no transitions from %v", from))
	}
	return t.Targets[from][sampleIndex(gen, t.Probs[from])]
}
