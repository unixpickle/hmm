package hmm

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&HMM{}).SerializerType(), DeserializeHMM)
}

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
	States  []State
	Emitter Emitter

	// TerminalState is the state that signals the end of
	// an observation chain.
	// Terminal states do not produce an emission, and are
	// implied by the end of the sequence.
	//
	// If TerminalState is nil, then probabilities are
	// computed without accounting for termination.
	TerminalState State

	// Init stores the initial state distribution.
	// It maps states to log probabilities.
	// If a state is absent, it has 0 probability.
	Init map[State]float64

	// Transitions stores the log probability for every
	// allowed transition.
	// If a transition is absent, it has 0 probability.
	Transitions map[Transition]float64
}

// DeserializeHMM deserializes an HMM.
func DeserializeHMM(d []byte) (h *HMM, err error) {
	defer essentials.AddCtxTo("deserialize HMM", &err)

	var terminalIdx int
	var states []serializer.Serializer
	var initStates []serializer.Serializer
	var initProbs []float64
	var transitionStates []serializer.Serializer
	var transitionProbs []float64
	var emitter Emitter
	err = serializer.DeserializeAny(d, &terminalIdx, &states, &initStates, &initProbs,
		&transitionStates, &transitionProbs, &emitter)
	if err != nil {
		return nil, err
	}
	if len(transitionStates)%2 != 0 || len(transitionProbs) != len(transitionStates)/2 ||
		len(initProbs) != len(initStates) || terminalIdx >= len(states) {
		return nil, errors.New("invalid slice size")
	} else if !serializersComparable(states, initStates, transitionStates) {
		return nil, errors.New("State or Obs not comparable")
	}
	h = &HMM{
		Emitter:     emitter,
		Init:        map[State]float64{},
		Transitions: map[Transition]float64{},
	}
	if terminalIdx >= 0 {
		h.TerminalState = states[terminalIdx]
	}
	for _, state := range states {
		h.States = append(h.States, state)
	}
	for i, prob := range transitionProbs {
		t := Transition{
			From: transitionStates[i*2],
			To:   transitionStates[i*2+1],
		}
		h.Transitions[t] = prob
	}
	for i, state := range initStates {
		h.Init[state] = initProbs[i]
	}
	return h, nil
}

// RandomHMM creates an HMM with discrete observations and
// random parameters.
// The resulting Emitter is a TabularEmitter.
//
// If gen is non-nil, it is used to generate all of the
// random parameters.
//
// RandomHMM may be used to generate starting points for
// BaumWelch.
func RandomHMM(gen *rand.Rand, states []State, terminal State, obs []Obs) *HMM {
	res := &HMM{
		States:        states,
		Emitter:       TabularEmitter{},
		TerminalState: terminal,
		Init:          map[State]float64{},
		Transitions:   map[Transition]float64{},
	}
	for i, prob := range randomDist(gen, len(states)) {
		res.Init[states[i]] = prob
	}
	emitter := res.Emitter.(TabularEmitter)
	for _, state := range states {
		if state == terminal {
			continue
		}
		for i, prob := range randomDist(gen, len(states)) {
			to := states[i]
			res.Transitions[Transition{From: state, To: to}] = prob
		}
		emitter[state] = map[Obs]float64{}
		for i, prob := range randomDist(gen, len(obs)) {
			emitter[state][obs[i]] = prob
		}
	}
	return res
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

// SampleLen is like Sample, but it limits the sampled
// sequence length.
// Unlike Sample, SampleLen can be used without a terminal
// state.
func (h *HMM) SampleLen(gen *rand.Rand, maxLen int) ([]State, []Obs) {
	var states []State
	var obs []Obs

	state := h.sampleStart(gen)

	var ts *transSampler
	for i := 0; i < maxLen && state != h.TerminalState; i++ {
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

// SerializerType returns the unique ID used to serialize
// an HMM with the serializer package.
func (h *HMM) SerializerType() string {
	return "github.com/unixpickle/hmm.HMM"
}

// Serialize serializes the HMM.
//
// This requires that the States and Emitter implement the
// serializer.Serializer interface.
func (h *HMM) Serialize() (data []byte, err error) {
	defer essentials.AddCtxTo("serialize HMM", &err)
	terminalIdx := -1
	var states []serializer.Serializer
	var initStates []serializer.Serializer
	var initProbs []float64
	var transitionStates []serializer.Serializer
	var transitionProbs []float64
	for i, state := range h.States {
		stateSer, ok := state.(serializer.Serializer)
		if !ok {
			return nil, fmt.Errorf("not a Serializer: %T", state)
		}
		states = append(states, stateSer)
		if state == h.TerminalState {
			terminalIdx = i
		}
	}
	for state, prob := range h.Init {
		initStates = append(initStates, state.(serializer.Serializer))
		initProbs = append(initProbs, prob)
	}
	for trans, prob := range h.Transitions {
		transitionStates = append(transitionStates, trans.From.(serializer.Serializer),
			trans.To.(serializer.Serializer))
		transitionProbs = append(transitionProbs, prob)
	}
	return serializer.SerializeAny(terminalIdx, states, initStates, initProbs,
		transitionStates, transitionProbs, h.Emitter)
}

type transSampler struct {
	Targets map[State][]State
	Probs   map[State][]float64
}

func newTransSampler(states []State, trans map[Transition]float64) *transSampler {
	res := &transSampler{
		Targets: map[State][]State{},
		Probs:   map[State][]float64{},
	}
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
